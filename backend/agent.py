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
        source = Storage.get_source(source_id)
        if not source:
            raise ValueError("Source video not found.")

        t = Storage.get_transcript(source_id)
        if not t or not t.get("segments"):
            raise ValueError("Transcript not ready for this source.")

        segments = t.get("segments", [])
        seg_lines = [f"[{s.get('timestamp_str', '00:00')}] {s['text']}" for s in segments]
        full_transcript = "\n".join(seg_lines)

        system_prompt = (
            "You are an elite professor writing definitive lecture notes for a single class.\n"
            f"Lecture Title: {source['title']}\n"
            f"Channel: {source.get('channel', 'YouTube')} | Duration: {int(source.get('duration', 0))}s\n\n"
            "Format requirements:\n"
            f"1. # {source['title']} - Detailed Lecture Notes\n"
            "2. > Executive Summary: 3-4 sentence core thesis and high-yield takeaway\n"
            "3. ## Deep-Dive Topic Breakdowns: Numbered conceptual sections explaining theories, algorithms, math equations ($$...$$), workflows, or code\n"
            "4. > Key Takeaways: Blockquote summary of essential exam/practical points\n"
            "5. ## Chronological Timestamp Guide: Key moments with exact [HH:MM:SS] timestamps\n"
            "Produce comprehensive, rigorous notes with zero fluff."
        )

        user_prompt = f"Transcript with timestamps:\n\n{full_transcript[:25000]}\n\nPlease write the complete lecture notes."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        md_content = groq_router.route_chat(messages, tier="heavy", temperature=0.2, max_tokens=4096)
        title = f"Lecture Notes: {source['title'][:40]}"

        tex_path = NoteExporter.markdown_to_latex(title, source.get("channel", "YouTube Instructor"), md_content)
        pdf_path = NoteExporter.markdown_to_pdf(title, source.get("channel", "YouTube Instructor"), md_content)

        artifact = Storage.save_artifact(
            notebook_id=notebook_id,
            source_id=source_id,
            title=title,
            type="lecture_note",
            content_md=md_content,
            content_tex=open(tex_path).read(),
            pdf_path=pdf_path,
            metadata={"source_title": source["title"], "source_id": source_id}
        )
        return artifact

    @staticmethod
    def generate_master_course_booklet(notebook_id: str) -> Dict[str, Any]:
        """
        Generates a unified master course textbook/booklet concatenating & synthesizing
        all playlist lectures into a single unified Markdown, LaTeX, and compiled PDF.
        Uses the 8 Groq keys in parallel for ultra-fast chapter generation!
        """
        nb = Storage.get_notebook(notebook_id)
        nb_title = nb["title"] if nb else "Complete Course Master Textbook"
        sources = Storage.get_sources_for_notebook(notebook_id)
        ready_sources = [s for s in sources if s.get("status") == "ready"]

        if not ready_sources:
            raise ValueError("No ready video sources found in this notebook/playlist.")

        logger.info(f"Generating Master Course Booklet across {len(ready_sources)} lectures in parallel...")

        # Helper to synthesize a single chapter in parallel via Groq router
        def synthesize_chapter(index_and_source: Tuple[int, Dict[str, Any]]) -> Tuple[int, str]:
            idx, src = index_and_source
            t = Storage.get_transcript(src["id"])
            segments = t.get("segments", []) if t else []
            sample_text = "\n".join([f"[{s.get('timestamp_str', '00:00')}] {s['text']}" for s in segments[:120]])

            sys_prompt = (
                f"You are a textbook author writing Chapter {idx + 1} of a master textbook.\n"
                f"Chapter Title: {src['title']}\n"
                "Format in Markdown:\n"
                f"## Chapter {idx + 1}: {src['title']}\n"
                "> Chapter Overview: High-level overview\n"
                "### Core Principles & Technical Mechanics: Detailed analysis with formulas and examples\n"
                "> Key Takeaways: 2-3 bullet summary\n"
            )
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Transcript:\n{sample_text}\n\nWrite Chapter {idx + 1}."}
            ]
            chapter_text = groq_router.route_chat(msgs, tier="heavy", temperature=0.2, max_tokens=3000)
            return idx, chapter_text

        # Execute chapter synthesis concurrently across all 8 Groq keys
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            chapter_results = list(executor.map(synthesize_chapter, enumerate(ready_sources)))

        # Sort chapters in original lecture order
        chapter_results.sort(key=lambda x: x[0])
        chapters_md = [c[1] for c in chapter_results]

        # Generate Master Syllabus Overview
        overview_prompt = [
            {"role": "system", "content": f"You are the lead author of the master textbook '{nb_title}'. Write an inspiring course preface, syllabus map, and cross-lecture concept relationship matrix."},
            {"role": "user", "content": f"Lectures in course:\n" + "\n".join([f"- Lecture {i+1}: {s['title']}" for i, s in enumerate(ready_sources)])}
        ]
        preface_md = groq_router.route_chat(overview_prompt, tier="heavy", temperature=0.3, max_tokens=2000)

        # Assemble Full Master Textbook
        book_lines = [
            f"# {nb_title} - Master Course Textbook",
            f"> **Comprehensive Unified Course Notes & Knowledge Base**",
            f"> *Total Lectures Ingested: {len(ready_sources)} | Generated by YouTube NotebookLM*",
            "",
            "---",
            "",
            preface_md,
            "",
            "---",
            ""
        ]
        book_lines.extend(chapters_md)

        full_book_md = "\n\n".join(book_lines)
        book_title = f"{nb_title} - Master Course Textbook"

        tex_path = NoteExporter.markdown_to_latex(book_title, "YouTube NotebookLM Master Series", full_book_md)
        pdf_path = NoteExporter.markdown_to_pdf(book_title, "YouTube NotebookLM Master Series", full_book_md)

        artifact = Storage.save_artifact(
            notebook_id=notebook_id,
            title=book_title,
            type="master_booklet",
            content_md=full_book_md,
            content_tex=open(tex_path).read(),
            pdf_path=pdf_path,
            metadata={"total_lectures": len(ready_sources), "scope": "full_course"}
        )

        return artifact

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
