import json
import logging
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple
from backend.groq_router import groq_router
from backend.storage import Storage
from backend.topic_indexer import TopicIndexer
from backend.exporters import NoteExporter

logger = logging.getLogger("ParallelSynthesizer")

class ParallelSynthesizer:
    @staticmethod
    def synthesize_master_booklet(notebook_id: str) -> Dict[str, Any]:
        """
        Map-Reduce High-Throughput Synthesis:
        1. Reads topic index across all sources in the notebook.
        2. Plans a Master Course Outline.
        3. Dispatches section synthesis in parallel across all 8 Groq keys.
        4. Compiles into Markdown, LaTeX (.tex), PDF (.pdf), and Standalone Web HTML (.html).
        """
        nb = Storage.get_notebook(notebook_id)
        nb_title = nb["title"] if nb else "Master Course Textbook"
        
        # 1. Gather all topics
        all_topics = Storage.get_notebook_topic_index(notebook_id)
        sources = Storage.get_sources_for_notebook(notebook_id)
        ready_sources = [s for s in sources if s.get("status") == "ready"]

        if not ready_sources:
            raise ValueError("No transcribed video sources found in this notebook.")

        # If topic index not yet populated for some sources, index them on the fly
        if not all_topics:
            for s in ready_sources:
                topics = TopicIndexer.index_source_topics(s["id"])
                all_topics.extend(topics)

        logger.info(f"⚡ [Parallel Synthesizer] Discovered {len(all_topics)} topics across {len(ready_sources)} sources.")

        # 2. Plan Master Outline (Table of Contents)
        outline_prompt = (
            f"You are the Editor-in-Chief designing the master textbook '{nb_title}'.\n"
            f"Here are the topics covered across the course lectures:\n"
            + "\n".join([f"- [{t['source_title']}] Topic: {t['title']} | Key idea: {t['summary']}" for t in all_topics[:40]])
            + "\n\nPlan 4 to 8 comprehensive CHAPTERS/SECTIONS for the master textbook.\n"
            "Return a valid JSON array of objects with:\n"
            "[\n"
            "  {\n"
            "    \"chapter_num\": 1,\n"
            "    \"chapter_title\": \"Chapter title\",\n"
            "    \"target_topics\": [\"topic title 1\", \"topic title 2\"],\n"
            "    \"focus_areas\": \"What specific mechanisms, formulas, or code this chapter must detail.\"\n"
            "  }\n"
            "]\n"
            "Return ONLY the JSON array without backticks or markdown fences."
        )

        try:
            raw_outline = groq_router.route_chat(
                [{"role": "user", "content": outline_prompt}],
                tier="fast",
                temperature=0.2,
                max_tokens=2000
            )
            clean_json = raw_outline.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
            outline = json.loads(clean_json)
        except Exception as e:
            logger.warning(f"Outline planning JSON parse error: {e}. Falling back to lecture-by-lecture outline.")
            outline = [
                {
                    "chapter_num": i + 1,
                    "chapter_title": s["title"],
                    "target_topics": [s["title"]],
                    "focus_areas": "Comprehensive breakdown of concepts and timestamps."
                }
                for i, s in enumerate(ready_sources)
            ]

        # 3. Parallel Chapter Synthesis across 8 Groq Keys
        def synthesize_section_worker(item: Dict[str, Any]) -> Tuple[int, str]:
            ch_num = item.get("chapter_num", 1)
            ch_title = item.get("chapter_title", f"Chapter {ch_num}")
            focus = item.get("focus_areas", "")
            target_topics = item.get("target_topics", [])

            # Pull exact transcript spans for matched topics
            relevant_spans = []
            for t in all_topics:
                # Match topic title or keywords
                is_match = any(tt.lower() in t["title"].lower() or t["title"].lower() in tt.lower() for tt in target_topics)
                if is_match or len(target_topics) == 0:
                    span_text = TopicIndexer.get_topic_transcript_span(t["source_id"], t["start_seconds"], t["end_seconds"])
                    if span_text:
                        relevant_spans.append(
                            f"--- Source: {t['source_title']} [{t['start_time']} - {t['end_time']}] ---\n{span_text}"
                        )

            if not relevant_spans and ready_sources:
                # Fallback to source transcript
                src = ready_sources[(ch_num - 1) % len(ready_sources)]
                t = Storage.get_transcript(src["id"])
                if t and t.get("segments"):
                    relevant_spans.append("\n".join([f"[{s.get('timestamp_str', '00:00')}] {s['text']}" for s in t["segments"][:100]]))

            context_blob = "\n\n".join(relevant_spans[:3])
            if len(context_blob) > 6000:
                context_blob = context_blob[:6000] + "\n...[transcript excerpt]"

            sys_prompt = (
                f"You are writing Chapter {ch_num} of the academic master textbook '{nb_title}'.\n"
                f"Chapter Title: {ch_title}\n"
                f"Focus: {focus}\n\n"
                "Requirements:\n"
                f"1. ## Chapter {ch_num}: {ch_title}\n"
                "2. > Chapter Overview: 2-3 sentence high-impact summary\n"
                "3. ### Deep-Dive Technical Breakdowns: Detailed conceptual, algorithmic, and mathematical explanations ($$...$$ or code blocks)\n"
                "4. > Key Takeaways: Bullet points of core formulas/rules\n"
                "5. Include exact timestamp references [HH:MM:SS] cited from the transcripts.\n"
                "Write in rigorous, publication-grade academic style with zero fluff."
            )

            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Transcript Evidence:\n\n{context_blob}\n\nWrite Chapter {ch_num} now."}
            ]

            logger.info(f"🚀 [Parallel Synthesizer] Key worker dispatching Chapter {ch_num}: '{ch_title}'")
            chapter_md = groq_router.route_chat(msgs, tier="heavy", temperature=0.2, max_tokens=2500)
            return ch_num, chapter_md

        # Execute all sections concurrently across 8 Groq keys
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            rendered_chapters = list(executor.map(synthesize_section_worker, outline))

        rendered_chapters.sort(key=lambda x: x[0])
        chapters_content = [c[1] for c in rendered_chapters]

        # 4. Generate Master Preface & Syllabus Matrix
        preface_prompt = [
            {"role": "system", "content": f"Write an academic preface, course syllabus overview, and concept roadmap for the master textbook '{nb_title}'."},
            {"role": "user", "content": f"Lectures Included:\n" + "\n".join([f"- {s['title']}" for s in ready_sources])}
        ]
        preface_md = groq_router.route_chat(preface_prompt, tier="heavy", temperature=0.3, max_tokens=1800)

        # 5. Assemble Master Document
        master_doc_lines = [
            f"# {nb_title}",
            f"> **Master Course Textbook & Comprehensive Lecture Compendium**",
            f"> *Total Ingested Lectures: {len(ready_sources)} | Generated via 8-Key Groq Parallel Engine*",
            "",
            "---",
            "",
            preface_md,
            "",
            "---",
            ""
        ]
        master_doc_lines.extend(chapters_content)

        full_master_md = "\n\n".join(master_doc_lines)
        book_title = f"{nb_title} - Master Course Textbook"

        # Generate All 4 Export Formats
        tex_path = NoteExporter.markdown_to_latex(book_title, "YouTube NotebookLM Master Series", full_master_md)
        pdf_path = NoteExporter.markdown_to_pdf(book_title, "YouTube NotebookLM Master Series", full_master_md)
        html_path = NoteExporter.markdown_to_standalone_html(book_title, "YouTube NotebookLM Master Series", full_master_md)

        artifact = Storage.save_artifact(
            notebook_id=notebook_id,
            title=book_title,
            type="master_booklet",
            content_md=full_master_md,
            content_tex=open(tex_path).read(),
            pdf_path=pdf_path,
            metadata={
                "total_chapters": len(rendered_chapters),
                "total_lectures": len(ready_sources),
                "html_path": html_path
            }
        )

        logger.info(f"🎉 Master Course Textbook compiled successfully! ({len(rendered_chapters)} chapters)")
        return artifact

    @staticmethod
    def synthesize_single_lecture(notebook_id: str, source_id: str) -> Dict[str, Any]:
        """
        Fast Single Lecture Parallel Synthesis:
        Extracts subtopics for that single video and writes an in-depth lecture booklet.
        """
        source = Storage.get_source(source_id)
        if not source:
            raise ValueError(f"Source {source_id} not found.")

        topics = Storage.get_source_topic_index(source_id)
        if not topics:
            topics = TopicIndexer.index_source_topics(source_id)

        t = Storage.get_transcript(source_id)
        if not t or not t.get("segments"):
            raise ValueError("Transcript not available for this source.")

        segments = t["segments"]
        transcript_text = "\n".join([f"[{s.get('timestamp_str', '00:00')}] {s['text']}" for s in segments])

        system_prompt = (
            "You are an elite academic professor writing definitive lecture notes.\n"
            f"Lecture Title: {source['title']}\n"
            f"Channel: {source.get('channel', 'YouTube')} | Duration: {int(source.get('duration', 0))}s\n\n"
            "Format in Markdown:\n"
            f"1. # {source['title']} - Comprehensive Lecture Notes\n"
            "2. > Executive Summary: 3-4 sentence core thesis\n"
            "3. ## Deep-Dive Topic Breakdowns: Numbered conceptual sections explaining theories, algorithms, math equations ($$...$$), workflows, or code\n"
            "4. > Key Takeaways: Blockquote summary of essential exam/practical points\n"
            "5. ## Chronological Timestamp Guide: Key milestones with exact [HH:MM:SS] timestamps\n"
            "Produce comprehensive, rigorous notes with zero fluff."
        )

        user_prompt = f"Transcript with timestamps:\n\n{transcript_text[:8000]}\n\nWrite the complete lecture notes."

        md_content = groq_router.route_chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            tier="heavy",
            temperature=0.2,
            max_tokens=2500
        )

        title = f"Lecture Notes: {source['title'][:40]}"
        author = source.get("channel", "YouTube Instructor")

        tex_path = NoteExporter.markdown_to_latex(title, author, md_content)
        pdf_path = NoteExporter.markdown_to_pdf(title, author, md_content)
        html_path = NoteExporter.markdown_to_standalone_html(title, author, md_content)

        artifact = Storage.save_artifact(
            notebook_id=notebook_id,
            source_id=source_id,
            title=title,
            type="lecture_note",
            content_md=md_content,
            content_tex=open(tex_path).read(),
            pdf_path=pdf_path,
            metadata={"source_title": source["title"], "source_id": source_id, "html_path": html_path}
        )

        return artifact
