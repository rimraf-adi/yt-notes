import sqlite3
import json
import uuid
import time
from typing import List, Dict, Any, Optional
from backend.config import DB_PATH

def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # Notebooks table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS notebooks (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    """)

    # Sources / Videos table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sources (
        id TEXT PRIMARY KEY,
        notebook_id TEXT NOT NULL,
        video_id TEXT,
        url TEXT NOT NULL,
        title TEXT NOT NULL,
        channel TEXT,
        duration REAL DEFAULT 0,
        thumbnail_url TEXT,
        status TEXT DEFAULT 'pending', -- pending, downloading, transcribing, ready, error
        progress REAL DEFAULT 0,
        error_message TEXT,
        audio_path TEXT,
        chapters_json TEXT DEFAULT '[]',
        created_at REAL NOT NULL,
        FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
    )
    """)

    # Transcripts table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transcripts (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL UNIQUE,
        full_text TEXT NOT NULL,
        segments_json TEXT NOT NULL, -- list of {start, end, text}
        created_at REAL NOT NULL,
        FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
    )
    """)

    # Topic Index table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS topic_index (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        notebook_id TEXT NOT NULL,
        topics_json TEXT NOT NULL, -- list of topic dicts
        created_at REAL NOT NULL,
        FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE,
        FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
    )
    """)

    # Studio Artifacts / Notes table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS artifacts (
        id TEXT PRIMARY KEY,
        notebook_id TEXT NOT NULL,
        source_id TEXT, -- optional, if note belongs to specific video or entire notebook
        title TEXT NOT NULL,
        type TEXT NOT NULL, -- comprehensive_notes, summary, study_guide, quiz, mindmap, podcast
        content_md TEXT NOT NULL,
        content_tex TEXT,
        pdf_path TEXT,
        metadata_json TEXT DEFAULT '{}',
        created_at REAL NOT NULL,
        FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
    )
    """)

    # Chat history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_messages (
        id TEXT PRIMARY KEY,
        notebook_id TEXT NOT NULL,
        role TEXT NOT NULL, -- user, assistant
        content TEXT NOT NULL,
        citations_json TEXT DEFAULT '[]',
        created_at REAL NOT NULL,
        FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
    )
    """)

    conn.commit()
    conn.close()

# Database Helper Functions
class Storage:
    @staticmethod
    def create_notebook(title: str = "My YouTube Notebook", description: str = "") -> Dict[str, Any]:
        conn = get_db()
        cursor = conn.cursor()
        nb_id = str(uuid.uuid4())
        now = time.time()
        cursor.execute(
            "INSERT INTO notebooks (id, title, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (nb_id, title, description, now, now)
        )
        conn.commit()
        conn.close()
        return Storage.get_notebook(nb_id)

    @staticmethod
    def get_notebooks() -> List[Dict[str, Any]]:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM notebooks ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def get_notebook(notebook_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM notebooks WHERE id = ?", (notebook_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def get_or_create_default_notebook() -> Dict[str, Any]:
        notebooks = Storage.get_notebooks()
        if notebooks:
            return notebooks[0]
        return Storage.create_notebook("YouTube Master Class Notebook", "Collection of notes and transcripts from YouTube")

    @staticmethod
    def add_source(
        notebook_id: str,
        url: str,
        title: str = "Loading video...",
        video_id: str = "",
        channel: str = "",
        duration: float = 0,
        thumbnail_url: str = "",
        chapters: List[Dict] = None
    ) -> Dict[str, Any]:
        conn = get_db()
        cursor = conn.cursor()
        source_id = str(uuid.uuid4())
        now = time.time()
        cursor.execute(
            """
            INSERT INTO sources (id, notebook_id, video_id, url, title, channel, duration, thumbnail_url, status, progress, chapters_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)
            """,
            (source_id, notebook_id, video_id, url, title, channel, duration, thumbnail_url, json.dumps(chapters or []), now)
        )
        conn.commit()
        conn.close()
        return Storage.get_source(source_id)

    @staticmethod
    def update_source_status(
        source_id: str,
        status: str,
        progress: float = 0,
        error_message: Optional[str] = None,
        title: Optional[str] = None,
        video_id: Optional[str] = None,
        channel: Optional[str] = None,
        duration: Optional[float] = None,
        thumbnail_url: Optional[str] = None,
        audio_path: Optional[str] = None,
        chapters: Optional[List[Dict]] = None
    ):
        conn = get_db()
        cursor = conn.cursor()
        
        updates = ["status = ?", "progress = ?"]
        params = [status, progress]

        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if video_id is not None:
            updates.append("video_id = ?")
            params.append(video_id)
        if channel is not None:
            updates.append("channel = ?")
            params.append(channel)
        if duration is not None:
            updates.append("duration = ?")
            params.append(duration)
        if thumbnail_url is not None:
            updates.append("thumbnail_url = ?")
            params.append(thumbnail_url)
        if audio_path is not None:
            updates.append("audio_path = ?")
            params.append(audio_path)
        if chapters is not None:
            updates.append("chapters_json = ?")
            params.append(json.dumps(chapters))

        params.append(source_id)
        cursor.execute(f"UPDATE sources SET {', '.join(updates)} WHERE id = ?", tuple(params))
        conn.commit()
        conn.close()

    @staticmethod
    def get_source(source_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sources WHERE id = ?", (source_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        res = dict(row)
        res["chapters"] = json.loads(res.get("chapters_json") or "[]")
        return res

    @staticmethod
    def get_sources_for_notebook(notebook_id: str) -> List[Dict[str, Any]]:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sources WHERE notebook_id = ? ORDER BY created_at ASC", (notebook_id,))
        rows = cursor.fetchall()
        conn.close()
        sources = []
        for r in rows:
            d = dict(r)
            d["chapters"] = json.loads(d.get("chapters_json") or "[]")
            sources.append(d)
        return sources

    @staticmethod
    def delete_source(source_id: str):
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        conn.commit()
        conn.close()

    @staticmethod
    def save_transcript(source_id: str, full_text: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        conn = get_db()
        cursor = conn.cursor()
        t_id = str(uuid.uuid4())
        now = time.time()
        cursor.execute(
            """
            INSERT OR REPLACE INTO transcripts (id, source_id, full_text, segments_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (t_id, source_id, full_text, json.dumps(segments), now)
        )
        conn.commit()
        conn.close()
        return {"id": t_id, "source_id": source_id, "full_text": full_text, "segments": segments}

    @staticmethod
    def get_transcript(source_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transcripts WHERE source_id = ?", (source_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        res = dict(row)
        res["segments"] = json.loads(res.get("segments_json") or "[]")
        return res

    @staticmethod
    def save_topic_index(source_id: str, notebook_id: str, topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        conn = get_db()
        cursor = conn.cursor()
        t_id = str(uuid.uuid4())
        now = time.time()
        # Delete existing index for this source if any
        cursor.execute("DELETE FROM topic_index WHERE source_id = ?", (source_id,))
        cursor.execute(
            "INSERT INTO topic_index (id, source_id, notebook_id, topics_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (t_id, source_id, notebook_id, json.dumps(topics), now)
        )
        conn.commit()
        conn.close()
        return {"id": t_id, "source_id": source_id, "notebook_id": notebook_id, "topics": topics}

    @staticmethod
    def get_source_topic_index(source_id: str) -> List[Dict[str, Any]]:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM topic_index WHERE source_id = ?", (source_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return []
        return json.loads(dict(row).get("topics_json") or "[]")

    @staticmethod
    def get_notebook_topic_index(notebook_id: str) -> List[Dict[str, Any]]:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM topic_index WHERE notebook_id = ? ORDER BY created_at ASC", (notebook_id,))
        rows = cursor.fetchall()
        conn.close()
        all_topics = []
        for r in rows:
            topics = json.loads(dict(r).get("topics_json") or "[]")
            all_topics.extend(topics)
        return all_topics

    @staticmethod
    def save_artifact(
        notebook_id: str,
        title: str,
        type: str,
        content_md: str,
        content_tex: str = "",
        pdf_path: str = "",
        source_id: Optional[str] = None,
        metadata: Dict = None
    ) -> Dict[str, Any]:
        conn = get_db()
        cursor = conn.cursor()
        art_id = str(uuid.uuid4())
        now = time.time()
        cursor.execute(
            """
            INSERT INTO artifacts (id, notebook_id, source_id, title, type, content_md, content_tex, pdf_path, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (art_id, notebook_id, source_id, title, type, content_md, content_tex, pdf_path, json.dumps(metadata or {}), now)
        )
        conn.commit()
        conn.close()
        return Storage.get_artifact(art_id)

    @staticmethod
    def get_artifact(art_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM artifacts WHERE id = ?", (art_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        res = dict(row)
        res["metadata"] = json.loads(res.get("metadata_json") or "{}")
        return res

    @staticmethod
    def get_artifacts_for_notebook(notebook_id: str) -> List[Dict[str, Any]]:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM artifacts WHERE notebook_id = ? ORDER BY created_at DESC", (notebook_id,))
        rows = cursor.fetchall()
        conn.close()
        artifacts = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d.get("metadata_json") or "{}")
            artifacts.append(d)
        return artifacts

    @staticmethod
    def add_chat_message(notebook_id: str, role: str, content: str, citations: List[Dict] = None) -> Dict[str, Any]:
        conn = get_db()
        cursor = conn.cursor()
        msg_id = str(uuid.uuid4())
        now = time.time()
        cursor.execute(
            "INSERT INTO chat_messages (id, notebook_id, role, content, citations_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (msg_id, notebook_id, role, content, json.dumps(citations or []), now)
        )
        conn.commit()
        conn.close()
        return {"id": msg_id, "notebook_id": notebook_id, "role": role, "content": content, "citations": citations or [], "created_at": now}

    @staticmethod
    def get_chat_history(notebook_id: str) -> List[Dict[str, Any]]:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chat_messages WHERE notebook_id = ? ORDER BY created_at ASC", (notebook_id,))
        rows = cursor.fetchall()
        conn.close()
        history = []
        for r in rows:
            d = dict(r)
            d["citations"] = json.loads(d.get("citations_json") or "[]")
            history.append(d)
        return history

# Initialize DB on load
init_db()
