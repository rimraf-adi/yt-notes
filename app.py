import os
import time
import json
import logging
import streamlit as st
from pathlib import Path
from datetime import datetime

# Antigravity / Backend Modules
from backend.config import BASE_DIR, DATA_DIR, EXPORTS_DIR, DOWNLOADS_DIR
from backend.storage import Storage
from backend.groq_router import groq_router
from backend.downloader import YouTubeDownloader
from backend.transcriber import Transcriber
from backend.agent import NotebookAgent
from backend.topic_indexer import TopicIndexer
from backend.parallel_synthesizer import ParallelSynthesizer
from backend.exporters import Exporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yt_notes_streamlit")

# Page Configuration - 100% Native Streamlit (No HTML / No CSS)
st.set_page_config(
    page_title="YouTube NotebookLM",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Persistent Session & Query Params -----------------
# 1. Check URL query params first, then session state, then SQLite database
active_nb_id = st.query_params.get("notebook_id") or st.session_state.get("notebook_id")
active_nb = Storage.get_notebook(active_nb_id) if active_nb_id else None

if not active_nb:
    # If no valid notebook is active, load the most recent notebook from SQLite
    all_nbs = Storage.list_notebooks()
    if all_nbs:
        active_nb = all_nbs[0]
    else:
        # Create initial notebook only if database is completely empty
        now_str = datetime.now().strftime("%I:%M %p")
        active_nb = Storage.create_notebook(f"Study Notebook ({now_str})")

notebook_id = active_nb["id"]
st.session_state.notebook_id = notebook_id
st.session_state.notebook_title = active_nb["title"]
st.query_params["notebook_id"] = notebook_id

# Restore current studio artifact if not in session state
if "current_artifact" not in st.session_state or st.session_state.current_artifact is None:
    saved_artifacts = Storage.get_notebook_artifacts(notebook_id)
    st.session_state.current_artifact = saved_artifacts[0] if saved_artifacts else None

# ----------------- Sidebar: Sources & Notebook Manager -----------------
with st.sidebar:
    st.title("🎓 YouTube NotebookLM")
    st.markdown(":green-background[⚡ 8-Key Groq Rotation Active]")
    st.divider()

    # Notebook Switcher / Creator
    col_nb1, col_nb2 = st.columns([3, 1])
    with col_nb1:
        st.subheader(f"📓 {st.session_state.get('notebook_title', 'Notebook')}")
    with col_nb2:
        if st.button("➕", help="Create new clean notebook"):
            now_str = datetime.now().strftime("%I:%M %p")
            new_nb = Storage.create_notebook(f"Study Notebook ({now_str})")
            st.session_state.notebook_id = new_nb["id"]
            st.session_state.notebook_title = new_nb["title"]
            st.session_state.chat_history = []
            st.session_state.current_artifact = None
            st.query_params["notebook_id"] = new_nb["id"]
            st.rerun()

    all_notebooks = Storage.list_notebooks()
    if len(all_notebooks) > 1:
        nb_options = {nb["id"]: nb["title"] for nb in all_notebooks}
        current_idx = list(nb_options.keys()).index(notebook_id) if notebook_id in nb_options else 0
        selected_nb_id = st.selectbox(
            "Switch Notebook",
            options=list(nb_options.keys()),
            format_func=lambda x: nb_options[x],
            index=current_idx,
            label_visibility="collapsed"
        )
        if selected_nb_id != notebook_id:
            st.session_state.notebook_id = selected_nb_id
            st.session_state.notebook_title = nb_options[selected_nb_id]
            st.session_state.chat_history = Storage.get_chat_history(selected_nb_id)
            artifacts = Storage.get_notebook_artifacts(selected_nb_id)
            st.session_state.current_artifact = artifacts[0] if artifacts else None
            st.query_params["notebook_id"] = selected_nb_id
            st.rerun()

    st.divider()
    st.subheader("📥 Ingest Videos / Playlists")
    
    url_input = st.text_input(
        "YouTube URL",
        placeholder="Paste YouTube Video or Playlist URL...",
        label_visibility="collapsed"
    )
    
    if st.button("🚀 Ingest & Transcribe", use_container_width=True):
        if url_input.strip():
            with st.spinner("Fetching video / playlist metadata..."):
                try:
                    videos = YouTubeDownloader.get_playlist_videos(url_input.strip())
                    st.info(f"Discovered {len(videos)} video(s). Registering in knowledge base...")
                    
                    # Pre-register all sources immediately
                    registered = []
                    for v in videos:
                        src = Storage.add_source(
                            notebook_id=notebook_id,
                            url=v["url"],
                            title=v.get("title", "Processing Video..."),
                            video_id=v.get("video_id", ""),
                            channel=v.get("channel", ""),
                            duration=v.get("duration", 0),
                            thumbnail_url=v.get("thumbnail_url", "")
                        )
                        registered.append(src)

                    # Process each source
                    prog_bar = st.progress(0, text="Starting ingestion pipeline...")
                    for idx, src in enumerate(registered):
                        src_title = src.get("title", "Video")
                        prog_bar.progress(int((idx / len(registered)) * 100), text=f"Processing ({idx+1}/{len(registered)}): {src_title[:30]}...")

                        vid_id = src.get("video_id", "") or YouTubeDownloader.extract_video_id(src.get("url", ""))

                        # ⚡ 1. Check if video was previously transcribed (Zero-Download Cache Hit)
                        cached_trans = Storage.get_transcript_by_video_id(vid_id)
                        if cached_trans:
                            logger.info(f"⚡ [Cache Hit] Reusing transcript for {vid_id} without downloading!")
                            Storage.save_transcript(src["id"], cached_trans["full_text"], cached_trans["segments"])
                            
                            # Copy cached topics
                            cached_topics = Storage.get_topic_index_by_video_id(vid_id)
                            if cached_topics:
                                for t in cached_topics:
                                    t["source_id"] = src["id"]
                                Storage.save_topic_index(src["id"], notebook_id, cached_topics)

                            Storage.update_source_status(
                                src["id"],
                                status="ready",
                                progress=100.0,
                                title=cached_trans.get("title") or src.get("title"),
                                duration=cached_trans.get("duration") or src.get("duration", 0),
                                channel=cached_trans.get("channel") or src.get("channel", "YouTube")
                            )
                            continue

                        # 2. Download audio if not in transcript cache
                        def update_dl_pct(p, msg):
                            pass
                        meta = YouTubeDownloader.download_audio(src["url"], src["id"], progress_callback=update_dl_pct)
                        Storage.update_source_status(
                            src["id"],
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

                        # 3. Transcribe with Whisper Large
                        Transcriber.process_source_audio(
                            source_id=src["id"],
                            audio_path=meta["audio_path"]
                        )

                        # 4. Topic Index
                        try:
                            TopicIndexer.index_source_topics(src["id"])
                        except Exception as te:
                            logger.warning(f"Topic indexing warning: {te}")

                        # 5. Clean up audio files from disk immediately to save space
                        YouTubeDownloader.cleanup_audio_files(src["id"], meta.get("audio_path"))

                        # 6. Mark ready
                        Storage.update_source_status(src["id"], status="ready", progress=100.0)

                    prog_bar.progress(100, text="✅ All videos ingested and indexed!")
                    time.sleep(1)
                    st.rerun()
                except Exception as ex:
                    st.error(f"Ingestion error: {ex}")
        else:
            st.warning("Please enter a valid YouTube URL.")

    st.divider()
    
    # List Sources
    sources = Storage.get_sources(notebook_id)
    ready_count = len([s for s in sources if s.get("status") == "ready"])
    st.subheader(f"📚 Sources ({ready_count}/{len(sources)} Ready)")
    
    if not sources:
        st.caption("No sources yet. Paste a YouTube URL above to begin.")
    else:
        for s in sources:
            thumb = s.get("thumbnail_url") or "https://images.unsplash.com/photo-1611162617213-7d7a39e9b1d7?w=160&q=80"
            dur_min = int(s.get("duration", 0) // 60)
            dur_sec = int(s.get("duration", 0) % 60)
            dur_str = f"{dur_min}m {dur_sec}s" if s.get("duration") else ""
            status = s.get("status", "pending")
            
            with st.expander(f"🎬 {s.get('title', 'Video')[:30]}...", expanded=False):
                st.image(thumb, use_container_width=True)
                st.caption(f"Channel: {s.get('channel', 'YouTube')} | Duration: {dur_str}")
                if status == "ready":
                    st.markdown(":green[✓ Ready]")
                elif status == "transcribing":
                    st.markdown(":blue[⏳ Transcribing...]")
                elif status == "error":
                    st.markdown(":red[⚠️ Error]")
                else:
                    st.caption("Queued")

                if st.button("🗑️ Delete", key=f"del_{s['id']}", use_container_width=True):
                    Storage.delete_source(s["id"])
                    st.rerun()

    # 8-Key Monitor in Sidebar
    st.divider()
    with st.expander("⚡ 8-Key Groq Pool Monitor", expanded=False):
        matrix_stats = groq_router.get_router_matrix_stats()
        st.caption(f"Total Keys: {matrix_stats.get('total_keys', 8)}")
        for k in matrix_stats.get("key_stats", []):
            st.markdown(f"**Key #{k['key_index']}** `{k['masked_key']}`: 🎙️ {k['transcriptions']} | 💬 {k['completions']}")

# ----------------- Main Screen: Tabs -----------------
st.title("YouTube NotebookLM")
st.caption("Grounded Agentic RAG • 8-Key Rotating Groq LLaMA 3.3 70B & DeepSeek R1 • LaTeX & PDF Publishing")

tab_chat, tab_studio, tab_transcripts = st.tabs([
    "💬 AI Chat & RAG",
    "📚 Studio Notes & Exporters",
    "🎙️ Transcripts & Topics"
])

# ----------------- TAB 1: Grounded RAG Chat -----------------
with tab_chat:
    # Display Chat Messages
    chat_history = Storage.get_chat_history(notebook_id)
    
    if not chat_history:
        st.info("💡 **Welcome to YouTube NotebookLM!** Ask any question across your ingested YouTube lectures and playlists. Every answer is grounded with clickable timestamp citations.")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            if st.button("📌 Executive Summary & Key Takeaways", use_container_width=True):
                st.session_state.pending_prompt = "Provide a comprehensive executive summary with key takeaways and principles from all ingested videos."
                st.rerun()
        with col_p2:
            if st.button("💻 Extract Formulas & Principles", use_container_width=True):
                st.session_state.pending_prompt = "Extract all mathematical formulas, core algorithms, and technical concepts mentioned across the lectures."
                st.rerun()
        with col_p3:
            if st.button("❓ 5-Question Active Recall Quiz", use_container_width=True):
                st.session_state.pending_prompt = "Generate a 5-question active recall test with solutions based on these video topics."
                st.rerun()
    else:
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            citations = msg.get("citations", [])
            with st.chat_message(role):
                st.markdown(content)
                if citations:
                    with st.expander(f"📌 {len(citations)} Source Citation(s)", expanded=False):
                        for c in citations:
                            st.markdown(f"**{c.get('source_title', 'Video')}** `[{c.get('timestamp_str', '00:00')}]`\n> {c.get('text', '')}")

    # Chat Input
    prompt_val = st.session_state.pop("pending_prompt", None)
    user_query = st.chat_input("Ask a question about your videos (e.g., 'Explain topic X with timestamps')...") or prompt_val

    if user_query:
        # Show User Message
        with st.chat_message("user"):
            st.markdown(user_query)
        Storage.add_chat_message(notebook_id, role="user", content=user_query)

        # Stream Assistant Response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                stream_gen, citations = NotebookAgent.answer_rag_stream(notebook_id, user_query)
                for chunk in stream_gen:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                
                if citations:
                    with st.expander(f"📌 {len(citations)} Source Citation(s)", expanded=True):
                        for c in citations:
                            st.markdown(f"**{c.get('source_title', 'Video')}** `[{c.get('timestamp_str', '00:00')}]`\n> {c.get('text', '')}")

                Storage.add_chat_message(notebook_id, role="assistant", content=full_response, citations=citations)
            except Exception as e:
                st.error(f"Error answering query: {e}")

# ----------------- TAB 2: Studio Notes & 4-Way Exporters -----------------
with tab_studio:
    st.subheader("🛠️ Study Studio & Multi-Format Exporters")
    st.caption("Generate publication-ready academic documents and download as **Markdown**, **Academic LaTeX**, **Compiled PDF**, or **Standalone HTML**.")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    with col_btn1:
        if st.button("⚡ Master Course Book (.PDF)", use_container_width=True, help="8-Key Parallel Map-Reduce across all playlist lectures"):
            with st.spinner("Synthesizing Master Course Textbook across all 8 Groq keys in parallel..."):
                try:
                    art = ParallelSynthesizer.synthesize_master_booklet(notebook_id)
                    st.session_state.current_artifact = art
                    st.success("Master Course Booklet synthesized!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Synthesis failed: {e}")
    with col_btn2:
        sources_ready = [s for s in Storage.get_sources(notebook_id) if s.get("status") == "ready"]
        if sources_ready:
            selected_source_for_note = st.selectbox("Select Lecture", options=sources_ready, format_func=lambda x: x["title"], label_visibility="collapsed")
            if st.button("📝 Single Lecture Note", use_container_width=True):
                with st.spinner(f"Generating Comprehensive Lecture Notes for {selected_source_for_note['title'][:25]}..."):
                    try:
                        art = ParallelSynthesizer.synthesize_single_lecture(notebook_id, selected_source_for_note["id"])
                        st.session_state.current_artifact = art
                        st.success("Lecture Note generated!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lecture note generation failed: {e}")
        else:
            st.button("📝 Single Lecture Note", disabled=True, use_container_width=True)
    with col_btn3:
        if st.button("🧠 Study Guide & Quiz", use_container_width=True):
            with st.spinner("Generating active recall study guide..."):
                try:
                    art = NotebookAgent.generate_study_guide(notebook_id)
                    st.session_state.current_artifact = art
                    st.success("Study Guide generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Study guide generation failed: {e}")
    with col_btn4:
        if st.button("🗺️ Concept Mind Map", use_container_width=True):
            with st.spinner("Building concept hierarchy..."):
                try:
                    art = NotebookAgent.generate_mindmap(notebook_id)
                    st.session_state.current_artifact = art
                    st.success("Mind Map generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Mind map generation failed: {e}")

    st.divider()

    # Render Current Artifact & Download Center
    artifacts = Storage.get_notebook_artifacts(notebook_id)
    if artifacts:
        current_art = st.session_state.get("current_artifact") or artifacts[0]
        
        st.subheader(f"📖 {current_art.get('title', 'Study Document')}")
        
        # 4 Export Download Buttons
        doc_title = current_art.get("title", "Study_Notes")
        md_content = current_art.get("content_md", "")

        exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
        
        # 1. Markdown
        with exp_col1:
            st.download_button(
                label="📄 Download .MD",
                data=md_content,
                file_name=f"{doc_title}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        # 2. Academic LaTeX
        with exp_col2:
            latex_content = Exporter.generate_latex(doc_title, md_content)
            st.download_button(
                label="📐 Download .TEX (LaTeX)",
                data=latex_content,
                file_name=f"{doc_title}.tex",
                mime="text/x-tex",
                use_container_width=True
            )

        # 3. PDF
        with exp_col3:
            try:
                pdf_filename = Exporter.generate_pdf(doc_title, md_content)
                pdf_path = EXPORTS_DIR / pdf_filename
                if pdf_path.exists():
                    with open(pdf_path, "rb") as pf:
                        st.download_button(
                            label="📕 Download .PDF",
                            data=pf.read(),
                            file_name=pdf_filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
            except Exception as pe:
                st.caption(f"PDF generation note: {pe}")

        # 4. Standalone HTML
        with exp_col4:
            html_content = Exporter.generate_html(doc_title, md_content)
            st.download_button(
                label="🌐 Download .HTML",
                data=html_content,
                file_name=f"{doc_title}.html",
                mime="text/html",
                use_container_width=True
            )

        st.divider()
        st.markdown(md_content)
    else:
        st.info("No study artifacts generated yet. Click any button above to synthesize your videos!")

# ----------------- TAB 3: Transcripts & Topics Index -----------------
with tab_transcripts:
    st.subheader("🎙️ Lecture Transcripts & Indexed Topic Boundaries")
    
    sources_with_transcripts = [s for s in Storage.get_sources(notebook_id) if s.get("status") == "ready"]
    if not sources_with_transcripts:
        st.info("Ingest and transcribe YouTube videos to inspect transcripts and extracted topic indices.")
    else:
        selected_source = st.selectbox(
            "Select Lecture Transcript",
            options=sources_with_transcripts,
            format_func=lambda x: f"{x['title']} ({int(x.get('duration', 0)//60)} mins)"
        )
        
        if selected_source:
            # 1. Structured Topic Map
            st.subheader("🗺️ Extracted Topic Boundaries")
            topics = Storage.get_source_topic_index(selected_source["id"])
            if topics:
                for t in topics:
                    with st.expander(f"📍 {t.get('title', 'Topic')} `[{t.get('start_time_str', '00:00')} - {t.get('end_time_str', '00:00')}]`", expanded=False):
                        st.markdown(f"**Summary:** {t.get('summary', '')}")
                        if t.get("key_takeaway"):
                            st.markdown(f"**Key Takeaway:** {t.get('key_takeaway')}")
                        if t.get("keywords"):
                            st.caption(f"Keywords: {', '.join(t.get('keywords', []))}")
            else:
                st.caption("Topic indexing in progress or empty for this source.")

            # 2. Raw Timestamped Transcript
            st.subheader("📜 Full Monotonic Transcript")
            trans = Storage.get_transcript(selected_source["id"])
            if trans and trans.get("segments"):
                for seg in trans["segments"]:
                    st.markdown(f"**`[{seg.get('timestamp_str', '00:00')}]`** {seg.get('text', '')}")
            else:
                st.caption("No transcript segments found.")
