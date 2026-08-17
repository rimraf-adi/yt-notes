#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "======================================================="
echo "   🚀 Starting YouTube NotebookLM (8-Key Groq Engine)  "
echo "======================================================="

# Verify .venv or create with uv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv
    uv pip install fastapi uvicorn python-dotenv groq yt-dlp pydub reportlab markdown jinja2 pydantic aiofiles httpx
fi

source .venv/bin/activate

echo "Starting server at: http://localhost:8000"
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
