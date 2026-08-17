import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import BASE_DIR, STATIC_DIR, EXPORTS_DIR, DOWNLOADS_DIR
from backend.storage import Storage
from backend.groq_router import groq_router
from backend.downloader import YouTubeDownloader
from backend.transcriber import Transcriber
from backend.agent import NotebookAgent
from backend.topic_indexer import TopicIndexer
from backend.parallel_synthesizer import ParallelSynthesizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yt_notes_app")

app = FastAPI(title="YouTube NotebookLM Clone", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class IngestRequest(BaseModel):
    notebook_id: str
    url: str

class ChatRequest(BaseModel):
    notebook_id: str
    query: str
    source_ids: Optional[List[str]] = None

class GenerateArtifactRequest(BaseModel):
    notebook_id: str
    type: str # comprehensive_notes, study_guide, mindmap, podcast
    source_id: Optional[str] = None

class CreateNotebookRequest(BaseModel):
    title: str
    description: Optional[str] = ""

# Background Worker for Ingestion
def process_single_video_task(source_id: str, url: str):
    """
    Background worker that downloads audio, chunks if necessary, transcribes with Whisper Large,
    and updates SQLite progress.
    """
    try:
        def update_progress(pct: float, msg: str):
            Storage.update_source_status(source_id, status="downloading" if pct < 55 else "transcribing", progress=pct)

        # 1. Download audio and extract metadata
        meta = YouTubeDownloader.download_audio(url, source_id, progress_callback=update_progress)
        
        Storage.update_source_status(
            source_id,
            status="transcribing",
            progress=55.0,
            title=meta["title"],
            video_id=meta["video_id"],
            channel=meta["channel"],
            duration=meta["duration"],
            thumbnail_url=meta["thumbnail_url"],
            audio_path=meta["audio_path"],
            chapters=meta["chapters"]
        )

        # 2. Check if this video_id was already transcribed in DB (Transcript Cache)
        conn = Storage.get_transcript(source_id) # check if existing
        existing_transcript = None
        
        # Look for any existing transcript for this video_id across all sources
        import sqlite3
        from backend.config import DB_PATH
        db_conn = sqlite3.connect(str(DB_PATH))
        db_conn.row_factory = sqlite3.Row
        cur = db_conn.cursor()
        cur.execute(
            """
            SELECT t.full_text, t.segments_json, s.id as prev_source_id 
            FROM transcripts t 
            JOIN sources s ON t.source_id = s.id 
            WHERE s.video_id = ? AND s.id != ? AND s.status = 'ready'
            LIMIT 1
            """,
            (meta["video_id"], source_id)
        )
        cached_row = cur.fetchone()
        db_conn.close()

        if cached_row:
            logger.info(f"⚡ [Transcript Cache Hit] Reusing existing transcript for {meta['video_id']}")
            import json
            segments = json.loads(cached_row["segments_json"])
            Storage.save_transcript(source_id, cached_row["full_text"], segments)
            
            # Copy topics
            prev_topics = Storage.get_source_topic_index(cached_row["prev_source_id"])
            if prev_topics:
                # Update source_id in topics
                for t in prev_topics:
                    t["source_id"] = source_id
                Storage.save_topic_index(source_id, Storage.get_source(source_id)["notebook_id"], prev_topics)
            
            Storage.update_source_status(source_id, status="ready", progress=100.0)
            logger.info(f"Source {source_id} instantly ready from cache!")
            return

        # 3. Transcribe audio with 8-key rotating Whisper Large
        def update_transcription_progress(pct: float, msg: str):
            Storage.update_source_status(source_id, status="transcribing", progress=pct)

        Transcriber.process_source_audio(
            source_id=source_id,
            audio_path=meta["audio_path"],
            progress_callback=update_transcription_progress
        )

        # 4. Automatically extract and index topics for precision RAG and parallel synthesis
        try:
            TopicIndexer.index_source_topics(source_id)
        except Exception as e:
            logger.warning(f"Topic indexing non-fatal error on {source_id}: {e}")

        # 5. Mark Ready
        Storage.update_source_status(source_id, status="ready", progress=100.0)
        logger.info(f"Source {source_id} ({meta['title']}) processed & topic-indexed successfully!")

    except Exception as e:
        logger.error(f"Failed processing source {source_id}: {e}", exc_info=True)
        Storage.update_source_status(source_id, status="error", progress=0, error_message=str(e))

def process_ingest_pipeline(notebook_id: str, url: str):
    """
    Detects if URL is a single video or playlist, registers sources, and processes sequentially.
    """
    try:
        # Check if playlist
        videos = YouTubeDownloader.get_playlist_videos(url)
        logger.info(f"Discovered {len(videos)} video(s) from URL: {url}")

        for v in videos:
            source = Storage.add_source(
                notebook_id=notebook_id,
                url=v["url"],
                title=v.get("title", "Processing Video..."),
                video_id=v.get("video_id", ""),
                channel=v.get("channel", ""),
                duration=v.get("duration", 0),
                thumbnail_url=v.get("thumbnail_url", "")
            )
            # Process synchronously in this background worker thread
            process_single_video_task(source["id"], v["url"])

    except Exception as e:
        logger.error(f"Failed during playlist/video ingestion: {e}", exc_info=True)

# API Endpoints
@app.get("/api/health")
def get_health():
    stats = groq_router.get_router_matrix_stats()
    return {
        "status": "healthy",
        "total_keys": stats["total_keys"],
        "active_keys": stats["active_keys"],
        "key_stats": stats["keys"],
        "supported_models": stats["supported_models"]
    }

@app.get("/api/notebooks")
def list_notebooks():
    return Storage.get_notebooks()

@app.post("/api/notebooks")
def create_notebook(req: CreateNotebookRequest):
    return Storage.create_notebook(req.title, req.description or "")

@app.get("/api/notebooks/{notebook_id}")
def get_notebook_details(notebook_id: str):
    nb = Storage.get_notebook(notebook_id)
    if not nb:
        raise HTTPException(status_code=404, detail="Notebook not found")
    sources = Storage.get_sources_for_notebook(notebook_id)
    artifacts = Storage.get_artifacts_for_notebook(notebook_id)
    chat_history = Storage.get_chat_history(notebook_id)
    return {
        "notebook": nb,
        "sources": sources,
        "artifacts": artifacts,
        "chat_history": chat_history
    }

@app.post("/api/sources/ingest")
def ingest_source(req: IngestRequest, bg_tasks: BackgroundTasks):
    nb = Storage.get_notebook(req.notebook_id)
    if not nb:
        nb = Storage.get_or_create_default_notebook()
    
    bg_tasks.add_task(process_ingest_pipeline, req.notebook_id, req.url)
    return {"message": "Ingestion job queued", "url": req.url}

@app.get("/api/sources/{source_id}")
def get_source_details(source_id: str):
    src = Storage.get_source(source_id)
    if not src:
        raise HTTPException(status_code=404, detail="Source not found")
    transcript = Storage.get_transcript(source_id)
    return {"source": src, "transcript": transcript}

@app.delete("/api/sources/{source_id}")
def delete_source(source_id: str):
    Storage.delete_source(source_id)
    return {"message": "Source deleted"}

@app.post("/api/chat")
def chat_with_notebook(req: ChatRequest):
    Storage.add_chat_message(req.notebook_id, role="user", content=req.query)
    stream_gen, citations = NotebookAgent.answer_rag_stream(req.notebook_id, req.query, req.source_ids)
    
    full_response = ""
    for chunk in stream_gen:
        full_response += chunk

    msg = Storage.add_chat_message(req.notebook_id, role="assistant", content=full_response, citations=citations)
    return msg

@app.get("/api/chat/stream")
def chat_stream(notebook_id: str = Query(...), query: str = Query(...)):
    """
    Server-Sent Events (SSE) streaming endpoint for real-time RAG answers.
    """
    Storage.add_chat_message(notebook_id, role="user", content=query)

    def event_generator():
        stream_gen, citations = NotebookAgent.answer_rag_stream(notebook_id, query)
        
        # Send citations first as metadata event
        yield f"event: citations\ndata: {json.dumps(citations)}\n\n"

        full_content = []
        for chunk in stream_gen:
            full_content.append(chunk)
            payload = json.dumps({"token": chunk})
            yield f"event: token\ndata: {payload}\n\n"

        complete_text = "".join(full_content)
        Storage.add_chat_message(notebook_id, role="assistant", content=complete_text, citations=citations)
        yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/artifacts/generate")
def generate_artifact(req: GenerateArtifactRequest):
    """
    Generates high-grade study artifacts with parallel 8-key map-reduce synthesis.
    """
    if req.type == "comprehensive_notes":
        if req.source_id:
            artifact = ParallelSynthesizer.synthesize_single_lecture(req.notebook_id, req.source_id)
        else:
            artifact = ParallelSynthesizer.synthesize_master_booklet(req.notebook_id)
    elif req.type == "study_guide":
        artifact = NotebookAgent.generate_study_guide(req.notebook_id)
    elif req.type == "mindmap":
        artifact = NotebookAgent.generate_mindmap(req.notebook_id)
    elif req.type == "podcast":
        artifact = NotebookAgent.generate_podcast_script(req.notebook_id)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown artifact type: {req.type}")

    return artifact

@app.get("/api/artifacts/{artifact_id}")
def get_artifact(artifact_id: str):
    art = Storage.get_artifact(artifact_id)
    if not art:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return art

@app.get("/api/exports/{filename}")
def download_export(filename: str):
    file_path = EXPORTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    return FileResponse(path=str(file_path), filename=filename)

@app.get("/api/audio/{filename}")
def stream_audio(filename: str):
    file_path = DOWNLOADS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=str(file_path), media_type="audio/mpeg")

# Mount Static Frontend
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")
