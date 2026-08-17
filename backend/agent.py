import json
import logging
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple, Iterator
from backend.groq_router import groq_router
from backend.rag_engine import RAGEngine
from backend.storage import Storage
from backend.exporters import NoteExporter
from backend.config import LLM_MODEL, REASONING_MODEL

logger = logging.getLogger(__name__)

class NotebookAgent:
    @staticmethod
    def get_notebook_full_context(notebook_id: str, source_id: Optional[str] = None) -> Tuple[str, str]:
        """
        Gathers complete or summarized transcript text and metadata for note generation.
        """
        if source_id:
            sources = [Storage.get_source(source_id)]
        else:
            sources = Storage.get_sources_for_notebook(notebook_id)

        valid_sources = [s for s in sources if s and s.get("status") == "ready"]
        if not valid_sources:
            return "", "No ready transcripts found in notebook."

        context_parts = []
        titles = []

        for s in valid_sources:
            t = Storage.get_transcript(s["id"])
            if not t:
                continue
            titles.append(f"• {s['title']} ({s.get('channel', 'YouTube')})")
            
            # Format with chapter markers and timestamped segments
            segments = t.get("segments", [])
            sample_rate = max(1, len(segments) // 80) if len(segments) > 120 else 1
            
            seg_lines = []
            for i in range(0, len(segments), sample_rate):
                seg = segments[i]
                seg_lines.append(f"[{seg.get('timestamp_str', '00:00')}] {seg['text']}")
            
            transcript_summary = "\n".join(seg_lines)
            context_parts.append(
                f"### VIDEO SOURCE: {s['title']}\n"
                f"Channel: {s.get('channel', 'YouTube')} | Duration: {int(s.get('duration', 0))}s\n"
                f"Transcript Highlights & Timestamps:\n{transcript_summary}\n"
            )

        combined_text = "\n\n".join(context_parts)
        sources_summary = "\n".join(titles)
        return combined_text, sources_summary

    @staticmethod
    def generate_single_lecture_note(notebook_id: str, source_id: str) -> Dict[str, Any]:
        """
        Generates in-depth, dedicated lecture notes, LaTeX, and PDF specifically for ONE video source.
        """
        from backend.parallel_synthesizer import ParallelSynthesizer
        return ParallelSynthesizer.synthesize_single_lecture(notebook_id, source_id)

    @staticmethod
    def generate_master_course_booklet(notebook_id: str) -> Dict[str, Any]:
        """
        Generates a unified master course textbook/booklet synthesizing
        all playlist lectures into a single unified Markdown, LaTeX, and compiled PDF
        using the 8 Groq keys in parallel with deep academic synthesis.
        """
        from backend.parallel_synthesizer import ParallelSynthesizer
        return ParallelSynthesizer.synthesize_master_booklet(notebook_id)

    @staticmethod
    def generate_comprehensive_notes(notebook_id: str, source_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Unified dispatch: if source_id is provided, generates single lecture note;
        otherwise generates comprehensive notes or master booklet.
        """
        if source_id:
            return NotebookAgent.generate_single_lecture_note(notebook_id, source_id)
        return NotebookAgent.generate_master_course_booklet(notebook_id)

    @staticmethod
    def generate_study_guide(notebook_id: str) -> Dict[str, Any]:
        """
        Generates Study Guide, Glossaries, and Practice Quizzes with Flashcards.
        """
        context, sources_summary = NotebookAgent.get_notebook_full_context(notebook_id)
        if not context:
            raise ValueError(sources_summary)

        system_prompt = (
            "You are an expert tutor creating a high-yield Study Guide and Active Recall Quiz from video material.\n"
            "Structure your output in Markdown with:\n"
            "1. # Study Guide & Active Recall Masterdeck\n"
            "2. ## High-Yield Concepts & Glossary (Key definitions)\n"
            "3. ## Core Q&A Drill (5 critical conceptual questions & answers)\n"
            "4. ## Multiple Choice Practice Quiz (4 questions with 4 options A-D, followed by an Answer Key with explanations)\n"
            "5. ## Flashcards Deck: Provide a list of 5-8 Q&A Flashcards for spaced repetition."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Sources:\n{context}\n\nGenerate the complete Study Guide and Quiz."}
        ]

        md_content = groq_router.route_chat(messages, tier="heavy", temperature=0.3, max_tokens=4096)
        title = "Study Guide & Active Recall Quiz"

        pdf_path = NoteExporter.markdown_to_pdf(title, "YouTube NotebookLM", md_content)

        artifact = Storage.save_artifact(
            notebook_id=notebook_id,
            title=title,
            type="study_guide",
            content_md=md_content,
            pdf_path=pdf_path,
            metadata={"sources": sources_summary}
        )
        return artifact

    @staticmethod
    def generate_mindmap(notebook_id: str) -> Dict[str, Any]:
        """
        Generates a Mermaid.js Mind Map representing topic architecture and relationships.
        """
        context, sources_summary = NotebookAgent.get_notebook_full_context(notebook_id)
        if not context:
            raise ValueError(sources_summary)

        system_prompt = (
            "You are an information architect. Create a comprehensive Mermaid.js diagram representing the concept hierarchy and flow of the material.\n"
            "Return ONLY valid Mermaid code wrapped in ```mermaid ... ```.\n"
            "Use either `graph TD` or `mindmap` syntax. Keep node labels short and concise without special characters that break Mermaid syntax."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Material:\n{context}\n\nGenerate the Mermaid mind map diagram."}
        ]

        mermaid_response = groq_router.route_chat(messages, tier="fast", temperature=0.2)

        md_content = f"# Concept Mind Map\n\nVisual overview of key themes and relationships:\n\n{mermaid_response}"
        title = "Concept Mind Map"

        artifact = Storage.save_artifact(
            notebook_id=notebook_id,
            title=title,
            type="mindmap",
            content_md=md_content,
            metadata={"mermaid": mermaid_response}
        )
        return artifact

    @staticmethod
    def generate_podcast_script(notebook_id: str) -> Dict[str, Any]:
        """
        Generates a NotebookLM-style 2-host audio podcast dialogue script (Alex & Maya).
        """
        context, sources_summary = NotebookAgent.get_notebook_full_context(notebook_id)
        if not context:
            raise ValueError(sources_summary)

        system_prompt = (
            "You are the creator of NotebookLM's famous Deep Dive audio overview podcast.\n"
            "Write an engaging, lively, insightful conversational dialogue between two hosts: Alex and Maya.\n"
            "They should break down the key ideas, debate trade-offs, use analogies, and discuss real-world implications of the video material in a natural conversational flow.\n\n"
            "Format:\n"
            "**Alex**: [dialogue]\n"
            "**Maya**: [dialogue]\n"
            "Include audio cues like [laughs], [thoughtful pause], [excitedly] sparingly for realism."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Video Material:\n{context}\n\nGenerate the complete 2-host podcast overview script."}
        ]

        md_content = groq_router.route_chat(messages, tier="heavy", temperature=0.6, max_tokens=4096)
        title = "Audio Deep Dive Podcast Script"

        pdf_path = NoteExporter.markdown_to_pdf(title, "YouTube NotebookLM", md_content)

        artifact = Storage.save_artifact(
            notebook_id=notebook_id,
            title=title,
            type="podcast",
            content_md=md_content,
            pdf_path=pdf_path
        )
        return artifact

    @staticmethod
    def answer_rag_stream(
        notebook_id: str,
        query: str,
        source_ids: Optional[List[str]] = None
    ) -> Tuple[Iterator[str], List[Dict[str, Any]]]:
        """
        Streams grounded RAG answer with inline timestamp citations.
        """
        context_str, citations = RAGEngine.build_rag_context(notebook_id, query, source_ids, top_k=6)
        
        # Fetch recent chat messages for multi-turn history
        history = Storage.get_chat_history(notebook_id)[-4:]
        
        system_prompt = (
            "You are YouTube NotebookLM AI, an intelligent grounded study assistant.\n"
            "Answer the user's question accurately based STRICTLY on the retrieved video transcripts provided below.\n"
            "Rules:\n"
            "1. When stating facts or quoting parts of the video, cite the source using inline markdown citations matching the source IDs, for example: `[1]` or `[2]`.\n"
            "2. If the user asks for timestamps or when something occurred, mention the timestamp in format `[HH:MM:SS]`.\n"
            "3. If the context does not contain the answer, politely state that it is not covered in the ingested videos.\n"
            "4. Keep answers articulate, well-structured, and helpful.\n\n"
            f"--- RETRIEVED SOURCES CONTEXT ---\n{context_str}\n---------------------------------"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": query})

        stream_gen = groq_router.route_chat_stream(messages, tier="heavy", temperature=0.3)
        return stream_gen, citations
