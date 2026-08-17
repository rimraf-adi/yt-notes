# 🎓 YouTube NotebookLM

> An AI-powered, high-throughput NotebookLM clone built specifically for YouTube playlists, lectures, and courses. Features an **8-Key Rotating Groq Engine** for Whisper Large transcription and parallel Map-Reduce synthesis into **Markdown, Academic LaTeX, Compiled PDF, and Web HTML**.

---

## ⚡ Key Highlights

- **8-Key Groq Key-Model Matrix Router**: Automatically load balances across 8 Groq API keys with thread-safe rate-limit tracking, cooldown management, and automatic model tier cascading (`llama-3.3-70b-versatile`, `deepseek-r1-distill-llama-70b`, `llama-3.1-8b-instant`).
- **`yt-dlp` Video & Playlist Ingestion**: Downloads 16kHz mono speech audio, chunks large videos (>1 hr) with continuous monotonic timestamp stitching, and extracts YouTube chapters.
- **Hierarchical Topic & Span Indexing**: Automatically parses transcripts into contiguous conceptual spans (`[start_seconds, end_seconds]`) to eliminate the "lost-in-the-middle" context window problem.
- **8-Key Concurrent Map-Reduce Synthesis**: Dispatches outline chapters concurrently across all 8 keys to generate comprehensive lecture booklets in seconds.
- **4 Export Formats**:
  1. 📥 **Markdown (`.md`)**
  2. 📐 **Academic LaTeX (`.tex`)**
  3. 📄 **Compiled PDF (`.pdf`)**
  4. 🌐 **Interactive Web Notes (`.html`)**
- **Grounded RAG Chat**: Inline clickable citations (`[04:20]`) that sync directly with the built-in audio player.

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10+
- `ffmpeg` installed on your system
- `uv` (recommended)

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/rimraf-adi/yt-notes.git
cd yt-notes

# Setup environment variables
cp .env.example .env
# Edit .env and paste your Groq API keys (comma-separated)
```

### 3. Run
```bash
chmod +x run.sh
./run.sh
```

Open your browser at: **http://localhost:8000**

---

## 🏗️ Architecture

```
yt-notes/
├── backend/
│   ├── config.py             # Configuration & path management
│   ├── groq_router.py        # 8-Key Matrix Router (High Throughput & Resilient Fallback)
│   ├── downloader.py         # yt-dlp audio extractor & ffmpeg chunker
│   ├── transcriber.py        # Whisper Large v3 transcription with timestamp stitching
│   ├── topic_indexer.py      # Hierarchical topic & span extraction
│   ├── parallel_synthesizer.py # 8-key parallel map-reduce synthesis
│   ├── rag_engine.py         # Sliding-window BM25 hybrid search
│   ├── storage.py            # SQLite database manager
│   ├── exporters.py          # MD, LaTeX, PDF, and HTML exporters
│   └── main.py               # FastAPI application with SSE streaming
├── frontend/
│   ├── index.html            # 3-Pane NotebookLM dark-mode interface
│   ├── style.css             # Glassmorphic UI & custom animations
│   └── app.js                # State management, SSE token streaming, player sync
├── run.sh                    # One-click startup script
└── README.md
```

---

## 📄 License
MIT
