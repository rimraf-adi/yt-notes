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
    uv pip install streamlit fastapi uvicorn python-dotenv groq yt-dlp pydub reportlab markdown jinja2 pydantic aiofiles httpx
fi

source .venv/bin/activate

echo "Starting YouTube NotebookLM Streamlit App at: http://localhost:8501"
streamlit run app.py --server.port 8501 --server.headless false
