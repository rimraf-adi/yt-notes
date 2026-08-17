import os
import re
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yt_dlp
from backend.config import DOWNLOADS_DIR
from backend.storage import Storage

logger = logging.getLogger(__name__)

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\/*?:"<>|]', "", name).replace(" ", "_")[:60]

class YouTubeDownloader:
    @staticmethod
    def extract_info(url: str, is_playlist_check: bool = True) -> Dict[str, Any]:
        """
        Extracts metadata for a single video or a playlist without downloading.
        """
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist" if is_playlist_check else False,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info

    @staticmethod
    def is_playlist(info: Dict[str, Any]) -> bool:
        return "_type" in info and info["_type"] == "playlist" or "entries" in info

    @staticmethod
    def get_playlist_videos(url: str) -> List[Dict[str, Any]]:
        """
        Returns list of basic video items in a playlist.
        """
        info = YouTubeDownloader.extract_info(url, is_playlist_check=True)
        if YouTubeDownloader.is_playlist(info):
            entries = info.get("entries", [])
            videos = []
            for e in entries:
                if not e:
                    continue
                v_id = e.get("id", "")
                v_url = e.get("url") or f"https://www.youtube.com/watch?v={v_id}"
                videos.append({
                    "video_id": v_id,
                    "url": v_url,
                    "title": e.get("title", "Untitled Video"),
                    "duration": e.get("duration", 0),
                    "channel": e.get("uploader") or e.get("channel", "Unknown Channel"),
                    "thumbnail_url": e.get("thumbnail") or (e.get("thumbnails", [{}])[-1].get("url") if e.get("thumbnails") else "")
                })
            return videos
        else:
            v_id = info.get("id", "")
            return [{
                "video_id": v_id,
                "url": url,
                "title": info.get("title", "Untitled Video"),
                "duration": info.get("duration", 0),
                "channel": info.get("uploader") or info.get("channel", "Unknown Channel"),
                "thumbnail_url": info.get("thumbnail") or ""
            }]

    @staticmethod
    def download_audio(url: str, source_id: str, progress_callback=None) -> Dict[str, Any]:
        """
        Downloads optimized audio (16kHz mono MP3) and extracts metadata + chapters.
        """
        output_template = str(DOWNLOADS_DIR / f"{source_id}_%(id)s.%(ext)s")
        
        def ydl_progress_hook(d):
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded = d.get("downloaded_bytes", 0)
                pct = (downloaded / total * 50.0) if total > 0 else 25.0
                if progress_callback:
                    progress_callback(pct, "Downloading audio...")
            elif d.get("status") == "finished":
                if progress_callback:
                    progress_callback(50.0, "Audio downloaded. Extracting...")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "64", # 64kbps is crystal clear for speech and very compact (~28MB for 1 hour)
                }
            ],
            "postprocessor_args": [
                "-ar", "16000",
                "-ac", "1"
            ],
            "progress_hooks": [ydl_progress_hook],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id", "")
            title = info.get("title", "YouTube Video")
            channel = info.get("uploader") or info.get("channel", "Unknown Channel")
            duration = info.get("duration", 0)
            thumbnail = info.get("thumbnail") or ""
            
            # Extract chapters if available
            chapters = []
            if "chapters" in info and info["chapters"]:
                for ch in info["chapters"]:
                    chapters.append({
                        "title": ch.get("title", "Chapter"),
                        "start_time": ch.get("start_time", 0),
                        "end_time": ch.get("end_time", 0)
                    })

            # Locate downloaded mp3 file
            audio_path = str(DOWNLOADS_DIR / f"{source_id}_{video_id}.mp3")
            if not os.path.exists(audio_path):
                # Look for matching file in DOWNLOADS_DIR
                for f in os.listdir(DOWNLOADS_DIR):
                    if f.startswith(source_id) and f.endswith(".mp3"):
                        audio_path = str(DOWNLOADS_DIR / f)
                        break

            return {
                "source_id": source_id,
                "video_id": video_id,
                "title": title,
                "channel": channel,
                "duration": duration,
                "thumbnail_url": thumbnail,
                "audio_path": audio_path,
                "chapters": chapters
            }

    @staticmethod
    def chunk_audio_if_needed(audio_path: str, max_size_mb: float = 23.0) -> List[Tuple[str, float]]:
        """
        Checks if audio file exceeds max_size_mb (Groq Whisper limit is 25MB).
        If so, splits audio into smaller MP3 chunks using ffmpeg.
        Returns a list of tuples: (chunk_file_path, start_offset_seconds).
        """
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb <= max_size_mb:
            return [(audio_path, 0.0)]

        logger.info(f"File {audio_path} size is {file_size_mb:.2f}MB > {max_size_mb}MB. Splitting into chunks...")
        
        # Get total duration using ffprobe
        duration_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ]
        try:
            total_duration = float(subprocess.check_output(duration_cmd).decode().strip())
        except Exception as e:
            logger.warning(f"ffprobe failed: {e}. Estimating duration from filesize.")
            total_duration = 3600 # fallback

        # Estimate chunk duration so each chunk is roughly 18MB
        chunk_duration = int(total_duration * (18.0 / file_size_mb))
        chunk_duration = max(300, min(chunk_duration, 1200)) # Between 5 mins and 20 mins
        overlap = 2 # 2 seconds overlap for Whisper context

        chunks = []
        start = 0.0
        chunk_idx = 0
        base_name = Path(audio_path).stem

        while start < total_duration:
            chunk_file = str(DOWNLOADS_DIR / f"{base_name}_part_{chunk_idx}.mp3")
            length = min(chunk_duration, total_duration - start)
            
            cmd = [
                "ffmpeg", "-y", "-ss", str(start), "-t", str(length),
                "-i", audio_path, "-acodec", "copy", chunk_file
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            chunks.append((chunk_file, start))
            
            start += (chunk_duration - overlap)
            chunk_idx += 1
            if length < chunk_duration:
                break

        return chunks
