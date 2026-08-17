import os
import json
import pytest
from pathlib import Path

from backend.config import DATA_DIR, EXPORTS_DIR, DOWNLOADS_DIR
from backend.storage import Storage, init_db
from backend.groq_router import groq_router
from backend.downloader import YouTubeDownloader
from backend.rag_engine import RAGEngine, TranscriptRAG
from backend.exporters import NoteExporter, Exporter
from backend.agent import NotebookAgent
from backend.topic_indexer import TopicIndexer
from backend.parallel_synthesizer import ParallelSynthesizer

@pytest.fixture(autouse=True)
def setup_database():
    init_db()

def test_storage_notebook_crud():
    # 1. Create Notebook
    nb = Storage.create_notebook("Test Algorithms Class", "Lecture notes on graph theory")
    assert nb is not None
    nb_id = nb["id"]
    assert nb["title"] == "Test Algorithms Class"

    # 2. List Notebooks
    notebooks = Storage.list_notebooks()
    assert len(notebooks) > 0
    assert any(n["id"] == nb_id for n in notebooks)

    # 3. Add Source
    source = Storage.add_source(
        notebook_id=nb_id,
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="Dijkstra Algorithm Lecture",
        video_id="dQw4w9WgXcQ",
        channel="MIT OpenCourseWare",
        duration=3600.0,
        thumbnail_url="https://example.com/thumb.jpg"
    )
    assert source is not None
    source_id = source["id"]

    # 4. Save Transcript
    sample_segments = [
        {"start": 0.0, "end": 15.0, "text": "Welcome to Dijkstra shortest path algorithm.", "timestamp_str": "00:00"},
        {"start": 15.0, "end": 45.0, "text": "We maintain a priority queue of minimum distances from the source vertex.", "timestamp_str": "00:15"},
        {"start": 45.0, "end": 90.0, "text": "The time complexity is O((V + E) log V) with a binary heap.", "timestamp_str": "00:45"}
    ]
    full_text = "Welcome to Dijkstra shortest path algorithm. We maintain a priority queue of minimum distances from the source vertex. The time complexity is O((V + E) log V) with a binary heap."
    t_res = Storage.save_transcript(source_id, full_text, sample_segments)
    assert t_res is not None

    fetched_t = Storage.get_transcript(source_id)
    assert fetched_t is not None
    assert len(fetched_t["segments"]) == 3

    # 5. Save Topic Index
    topics = [
        {
            "title": "Priority Queue Initialization",
            "start_time": 0.0,
            "end_time": 45.0,
            "start_time_str": "00:00",
            "end_time_str": "00:45",
            "summary": "Initializes distances and priority queue for shortest paths.",
            "key_takeaway": "All non-source vertices start with infinity distance.",
            "keywords": ["dijkstra", "priority queue", "graph"]
        },
        {
            "title": "Complexity Analysis",
            "start_time": 45.0,
            "end_time": 90.0,
            "start_time_str": "00:45",
            "end_time_str": "01:30",
            "summary": "Analyzes edge relaxation and heap operations.",
            "key_takeaway": "O((V + E) log V) time complexity.",
            "keywords": ["complexity", "binary heap", "relaxation"]
        }
    ]
    Storage.save_topic_index(source_id, nb_id, topics)
    fetched_topics = Storage.get_source_topic_index(source_id)
    assert len(fetched_topics) == 2

    # 6. Save & Fetch Artifact
    art = Storage.save_artifact(
        notebook_id=nb_id,
        title="Dijkstra Shortest Path Master Notes",
        type="comprehensive_notes",
        content_md="# Dijkstra Algorithm\n\nPriority queue relaxation logic.",
        source_id=source_id
    )
    assert art is not None
    artifacts = Storage.get_notebook_artifacts(nb_id)
    assert len(artifacts) >= 1
    assert artifacts[0]["title"] == "Dijkstra Shortest Path Master Notes"

    # 7. Add Chat Message
    msg = Storage.add_chat_message(
        notebook_id=nb_id,
        role="user",
        content="What is the time complexity of Dijkstra?",
        citations=[{"source_title": "Dijkstra Lecture", "timestamp_str": "00:45", "text": "O((V+E) log V)"}]
    )
    assert msg is not None
    chat_hist = Storage.get_chat_history(nb_id)
    assert len(chat_hist) >= 1

def test_exporters_all_four_formats(tmp_path):
    sample_title = "Graph Theory & Minimum Spanning Trees"
    sample_md = """# Graph Theory & Minimum Spanning Trees

## Executive Summary
This document explores Prim's and Kruskal's algorithms for finding minimum spanning trees.

### Key Equations
$$\\sum_{e \\in T} w(e) \\le \\sum_{e \\in T'} w(e)$$

### Algorithm Steps
1. Sort all edges in non-decreasing order.
2. Pick smallest edge using Union-Find.
3. Stop when $V-1$ edges are added.

```python
def kruskal(vertices, edges):
    edges.sort(key=lambda x: x[2])
    # Union-Find logic here
```
"""
    # 1. Markdown Export
    md_path = Exporter.export_to_markdown(sample_title, sample_md)
    assert os.path.exists(md_path)
    with open(md_path, "r", encoding="utf-8") as f:
        assert "# Graph Theory" in f.read()

    # 2. LaTeX Export
    tex_str = Exporter.generate_latex(sample_title, sample_md)
    assert "\\documentclass" in tex_str
    assert "\\section{Executive Summary}" in tex_str

    # 3. PDF Export
    pdf_filename = Exporter.generate_pdf(sample_title, sample_md)
    pdf_full_path = EXPORTS_DIR / pdf_filename
    assert pdf_full_path.exists()
    assert pdf_full_path.stat().st_size > 500 # Valid non-empty PDF

    # 4. Standalone HTML Export
    html_str = Exporter.generate_html(sample_title, sample_md)
    assert "<!DOCTYPE html>" in html_str
    assert "Graph Theory" in html_str

def test_rag_chunking_and_search():
    sample_segments = [
        {"start": 0.0, "end": 10.0, "text": "In computer science, dynamic programming is a method for solving complex problems.", "timestamp_str": "00:00"},
        {"start": 10.0, "end": 20.0, "text": "It breaks down problems into simpler subproblems recursively.", "timestamp_str": "00:10"},
        {"start": 20.0, "end": 30.0, "text": "Memozation and tabulation are two standard techniques.", "timestamp_str": "00:20"},
        {"start": 30.0, "end": 40.0, "text": "The Bellman-Ford algorithm uses dynamic programming for shortest paths with negative weights.", "timestamp_str": "00:30"},
        {"start": 40.0, "end": 50.0, "text": "It detects negative cycles in O(V * E) time.", "timestamp_str": "00:40"}
    ]

    chunks = RAGEngine.chunk_transcript(
        segments=sample_segments,
        chunk_window_sec=25.0,
        overlap_sec=5.0
    )
    assert len(chunks) >= 1
    assert "start" in chunks[0]
    assert "text" in chunks[0]

def test_groq_router_matrix_stats():
    stats = groq_router.get_router_matrix_stats()
    assert "total_keys" in stats
    assert stats["total_keys"] >= 1
    assert "key_stats" in stats
    assert len(stats["key_stats"]) == stats["total_keys"]

def test_audio_cleanup_mechanism(tmp_path):
    dummy_source_id = "test-cleanup-123"
    dummy_file = DOWNLOADS_DIR / f"{dummy_source_id}_dQw4w9WgXcQ.mp3"
    with open(dummy_file, "w") as f:
        f.write("dummy audio content")
    assert dummy_file.exists()

    YouTubeDownloader.cleanup_audio_files(dummy_source_id, str(dummy_file))
    assert not dummy_file.exists()

def test_groq_router_live_inference():
    response = groq_router.route_chat_completion(
        messages=[{"role": "user", "content": "Respond with the single word: READY"}],
        tier="fast",
        temperature=0.0
    )
    assert response is not None
    assert len(response.strip()) > 0


