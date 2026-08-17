import json
import logging
import re
from typing import List, Dict, Any, Optional
from backend.groq_router import groq_router
from backend.storage import Storage

logger = logging.getLogger("TopicIndexer")

class TopicIndexer:
    @staticmethod
    def index_source_topics(source_id: str) -> List[Dict[str, Any]]:
        """
        Reads a video transcript and uses Groq fast tier to extract a structured,
        timestamp-bounded Topic Index map for precision RAG and parallel synthesis.
        """
        source = Storage.get_source(source_id)
        if not source:
            logger.warning(f"Source {source_id} not found for indexing.")
            return []

        transcript = Storage.get_transcript(source_id)
        if not transcript or not transcript.get("segments"):
            logger.warning(f"No transcript segments found for source {source_id}.")
            return []

        segments = transcript["segments"]
        
        # Build condensed timestamped transcript
        sample_step = max(1, len(segments) // 100) if len(segments) > 120 else 1
        lines = []
        for i in range(0, len(segments), sample_step):
            s = segments[i]
            lines.append(f"[{s.get('timestamp_str', '00:00')}] (t={int(s.get('start', 0))}s) {s['text']}")
        condensed_transcript = "\n".join(lines[:120])

        system_prompt = (
            "You are an AI Curriculum & Knowledge Indexer.\n"
            "Analyze the following video transcript and divide it into 3 to 7 distinct, contiguous TOPICS.\n"
            "For each topic, return a valid JSON array of objects with the following keys:\n"
            "[\n"
            "  {\n"
            "    \"title\": \"Clear descriptive topic name\",\n"
            "    \"summary\": \"1-2 sentence core concept explanation\",\n"
            "    \"keywords\": [\"keyword1\", \"keyword2\", \"formula_or_code\"],\n"
            "    \"start_time\": \"HH:MM:SS\",\n"
            "    \"end_time\": \"HH:MM:SS\",\n"
            "    \"start_seconds\": 0.0,\n"
            "    \"end_seconds\": 180.0,\n"
            "    \"key_takeaway\": \"Main point from this section\"\n"
            "  }\n"
            "]\n"
            "Return ONLY the JSON array without backticks or markdown fences."
        )

        user_prompt = (
            f"Video Title: {source['title']}\n"
            f"Channel: {source.get('channel', 'YouTube')}\n\n"
            f"Transcript with timestamps:\n{condensed_transcript}\n\n"
            "Extract the topic index JSON array now:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            raw_response = groq_router.route_chat(messages, tier="fast", temperature=0.1, max_tokens=2500)
            
            # Clean JSON response
            clean_json = raw_response.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()

            topics_data = json.loads(clean_json)
            
            # Enrich topics with source metadata
            enriched_topics = []
            for i, t in enumerate(topics_data):
                t_obj = {
                    "topic_id": f"{source_id}_t{i+1}",
                    "source_id": source_id,
                    "source_title": source["title"],
                    "video_id": source.get("video_id", ""),
                    "channel": source.get("channel", "YouTube"),
                    "title": t.get("title", f"Topic {i+1}"),
                    "summary": t.get("summary", ""),
                    "keywords": t.get("keywords", []),
                    "start_time": t.get("start_time", "00:00"),
                    "end_time": t.get("end_time", "00:00"),
                    "start_seconds": float(t.get("start_seconds", 0.0)),
                    "end_seconds": float(t.get("end_seconds", 0.0)),
                    "key_takeaway": t.get("key_takeaway", "")
                }
                enriched_topics.append(t_obj)

            Storage.save_topic_index(source_id, source["notebook_id"], enriched_topics)
            logger.info(f"Indexed {len(enriched_topics)} topics for source {source_id} ('{source['title']}')")
            return enriched_topics

        except Exception as e:
            logger.error(f"Failed extracting topic index for {source_id}: {e}", exc_info=True)
            # Fallback: create single top-level topic
            fallback_topic = [{
                "topic_id": f"{source_id}_t1",
                "source_id": source_id,
                "source_title": source["title"],
                "video_id": source.get("video_id", ""),
                "channel": source.get("channel", "YouTube"),
                "title": source["title"],
                "summary": "Full video overview",
                "keywords": [source["title"]],
                "start_time": "00:00",
                "end_time": "00:00",
                "start_seconds": 0.0,
                "end_seconds": float(source.get("duration", 0.0)),
                "key_takeaway": "Comprehensive overview."
            }]
            Storage.save_topic_index(source_id, source["notebook_id"], fallback_topic)
            return fallback_topic

    @staticmethod
    def get_topic_transcript_span(source_id: str, start_sec: float, end_sec: float) -> str:
        """
        Retrieves the exact transcript lines that occur within [start_sec, end_sec].
        """
        t = Storage.get_transcript(source_id)
        if not t or not t.get("segments"):
            return ""

        matching_lines = []
        for s in t["segments"]:
            s_start = s.get("start", 0.0)
            s_end = s.get("end", 0.0)
            # Check overlap
            if (s_end >= start_sec) and (s_start <= end_sec):
                matching_lines.append(f"[{s.get('timestamp_str', '00:00')}] {s['text']}")

        if not matching_lines:
            # Fallback: take first 20 lines
            matching_lines = [f"[{s.get('timestamp_str', '00:00')}] {s['text']}" for s in t["segments"][:20]]

        return "\n".join(matching_lines)
