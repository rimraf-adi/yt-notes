import re
import math
from typing import List, Dict, Any, Optional, Tuple
from backend.storage import Storage

class RAGEngine:
    @staticmethod
    def chunk_transcript(segments: List[Dict[str, Any]], chunk_window_sec: float = 60.0, overlap_sec: float = 15.0) -> List[Dict[str, Any]]:
        """
        Groups small Whisper segments into contextual chunks (~45-60 seconds each)
        with preserved start/end timestamps and formatted timestamp labels.
        """
        if not segments:
            return []

        chunks = []
        current_chunk_segments = []
        chunk_start = segments[0]["start"]

        for seg in segments:
            current_chunk_segments.append(seg)
            chunk_end = seg["end"]

            if (chunk_end - chunk_start) >= chunk_window_sec:
                text = " ".join([s["text"] for s in current_chunk_segments]).strip()
                chunks.append({
                    "start": chunk_start,
                    "end": chunk_end,
                    "timestamp_str": current_chunk_segments[0].get("timestamp_str", "00:00"),
                    "text": text
                })
                # Slide window with overlap
                retained = []
                for s in current_chunk_segments:
                    if s["end"] >= (chunk_end - overlap_sec):
                        retained.append(s)
                current_chunk_segments = retained
                chunk_start = current_chunk_segments[0]["start"] if current_chunk_segments else seg["start"]

        if current_chunk_segments:
            text = " ".join([s["text"] for s in current_chunk_segments]).strip()
            chunks.append({
                "start": chunk_start,
                "end": current_chunk_segments[-1]["end"],
                "timestamp_str": current_chunk_segments[0].get("timestamp_str", "00:00"),
                "text": text
            })

        return chunks

    @staticmethod
    def search_transcripts(
        notebook_id: str,
        query: str,
        source_ids: Optional[List[str]] = None,
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Performs hybrid keyword & term-frequency search over transcript chunks
        for sources in a notebook.
        """
        sources = Storage.get_sources_for_notebook(notebook_id)
        if source_ids:
            sources = [s for s in sources if s["id"] in source_ids]

        query_terms = [re.sub(r'[^a-zA-Z0-9]', '', t.lower()) for t in query.split() if len(t) > 2]
        if not query_terms:
            query_terms = [query.lower()]

        all_chunks = []
        for src in sources:
            t = Storage.get_transcript(src["id"])
            if not t:
                continue
            
            chunks = RAGEngine.chunk_transcript(t.get("segments", []))
            for c in chunks:
                all_chunks.append({
                    "source_id": src["id"],
                    "source_title": src["title"],
                    "video_id": src.get("video_id", ""),
                    "start": c["start"],
                    "end": c["end"],
                    "timestamp_str": c["timestamp_str"],
                    "text": c["text"]
                })

        if not all_chunks:
            return []

        # Score chunks with BM25 / TF-IDF style metric
        scored_chunks = []
        for chunk in all_chunks:
            text_lower = chunk["text"].lower()
            score = 0.0
            
            # Direct phrase match bonus
            if query.lower() in text_lower:
                score += 10.0

            # Term frequency & proximity
            for term in query_terms:
                count = text_lower.count(term)
                if count > 0:
                    score += (1.0 + math.log(count)) * 2.0
                    # Title match bonus
                    if term in chunk["source_title"].lower():
                        score += 3.0

            if score > 0:
                scored_chunks.append((score, chunk))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        results = [c for _, c in scored_chunks[:top_k]]

        # If few keyword matches, return first chunks as fallback context
        if not results and all_chunks:
            results = all_chunks[:top_k]

        return results

    @staticmethod
    def build_rag_context(
        notebook_id: str,
        query: str,
        source_ids: Optional[List[str]] = None,
        top_k: int = 8
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Builds grounded context block and citation metadata for LLM prompt.
        """
        chunks = RAGEngine.search_transcripts(notebook_id, query, source_ids, top_k=top_k)
        
        context_blocks = []
        citations = []

        for i, c in enumerate(chunks, 1):
            citation_id = f"[{i}]"
            context_blocks.append(
                f"Source {citation_id}: \"{c['source_title']}\" at {c['timestamp_str']}\n"
                f"Content: {c['text']}\n"
            )
            citations.append({
                "citation_id": citation_id,
                "source_id": c["source_id"],
                "source_title": c["source_title"],
                "video_id": c["video_id"],
                "timestamp_str": c["timestamp_str"],
                "start_seconds": c["start"],
                "text_snippet": c["text"][:160] + "..."
            })

        formatted_context = "\n---\n".join(context_blocks)
        return formatted_context, citations

# Alias for compatibility
TranscriptRAG = RAGEngine
