import logging
from typing import List, Dict, Any, Callable, Optional
from backend.groq_router import groq_router
from backend.downloader import YouTubeDownloader
from backend.storage import Storage

logger = logging.getLogger(__name__)

def format_timestamp(seconds: float) -> str:
    """Converts seconds into HH:MM:SS or MM:SS format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

class Transcriber:
    @staticmethod
    def process_source_audio(
        source_id: str,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Splits audio if needed, transcribes all parts via rotating Groq Whisper Large keys,
        stitches timestamps accurately across chunks, and saves into database.
        """
        if progress_callback:
            progress_callback(55.0, "Preparing audio for Groq Whisper...")

        # 1. Chunk audio if it exceeds Groq size limit
        chunks = YouTubeDownloader.chunk_audio_if_needed(audio_path, max_size_mb=23.0)
        num_chunks = len(chunks)
        
        all_segments = []
        full_text_parts = []
        
        logger.info(f"Processing source {source_id} with {num_chunks} audio chunk(s)...")

        for idx, (chunk_file, time_offset) in enumerate(chunks):
            current_pct = 60.0 + (idx / num_chunks) * 35.0
            if progress_callback:
                progress_callback(current_pct, f"Transcribing audio (Part {idx + 1}/{num_chunks})...")

            # Transcribe with Whisper Large v3 via Key-Model Router
            result = groq_router.route_transcription(chunk_file)
            
            chunk_text = result.get("text", "").strip()
            raw_segments = result.get("segments", [])

            if raw_segments:
                for seg in raw_segments:
                    # Adjust timestamp by time_offset
                    start = seg.get("start", 0.0) + time_offset
                    end = seg.get("end", 0.0) + time_offset
                    text = seg.get("text", "").strip()
                    if text:
                        all_segments.append({
                            "start": round(start, 2),
                            "end": round(end, 2),
                            "timestamp_str": format_timestamp(start),
                            "text": text
                        })
            elif chunk_text:
                # If segments not provided in fallback, make single segment
                all_segments.append({
                    "start": round(time_offset, 2),
                    "end": round(time_offset + 300, 2),
                    "timestamp_str": format_timestamp(time_offset),
                    "text": chunk_text
                })

            full_text_parts.append(chunk_text)

        full_text = " ".join(full_text_parts)

        # 2. Save transcript to database
        saved = Storage.save_transcript(source_id, full_text, all_segments)
        
        if progress_callback:
            progress_callback(100.0, "Transcription complete!")

        return saved
