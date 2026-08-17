import os
import time
import json
import logging
import streamlit as st
from pathlib import Path
from datetime import datetime

# Antigravity / Backend Modules
from backend.config import BASE_DIR, DATA_DIR, EXPORTS_DIR, DOWNLOADS_DIR
from backend.storage import Storage
from backend.groq_router import groq_router
from backend.downloader import YouTubeDownloader
from backend.transcriber import Transcriber
from backend.agent import NotebookAgent
from backend.topic_indexer import TopicIndexer
from backend.parallel_synthesizer import ParallelSynthesizer
from backend.exporters import Exporter, NoteExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yt_notes_streamlit")

# Page Configuration - Clean Modern Architecture
st.set_page_config(
    page_title="YouTube NotebookLM",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- View State & Persistent Query Params -----------------
param_nb_id = st.query_params.get("notebook_id")
all_notebooks = Storage.list_notebooks()

if "view_mode" not in st.session_state:
    if param_nb_id and Storage.get_notebook(param_nb_id):
        st.session_state.view_mode = "studio"
        st.session_state.notebook_id = param_nb_id
    else:
        st.session_state.view_mode = "gallery"
        st.session_state.notebook_id = None

# =========================================================================
# 🏠 1. GALLERY VIEW (Home Dashboard)
# =========================================================================
if st.session_state.get("view_mode") == "gallery":
    st.title("🎓 YouTube NotebookLM")
    st.caption("Grounded Academic Research Engine • Multi-Key Parallel Groq Matrix • LaTeX & PDF Publishing")

    # Top Action Bar
    col_header, col_action = st.columns([3, 1])
    with col_header:
        st.subheader("📚 Research Notebooks")
    with col_action:
        if st.button("➕ Create New Notebook", type="primary"):
            st.session_state.show_create_modal = True
            st.rerun()

    # Create Modal / Container
    if st.session_state.get("show_create_modal", False):
        with st.container(border=True):
            st.markdown("#### ➕ Create New Notebook")
            nb_name_input = st.text_input("Notebook Title", placeholder="e.g. Yale Philosophy 176 - Death (Shelly Kagan)", key="create_nb_name")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if st.button("Create & Open", type="primary"):
                    title = nb_name_input.strip() if nb_name_input.strip() else f"Research Notebook ({datetime.now().strftime('%b %d, %I:%M %p')})"
                    new_nb = Storage.create_notebook(title)
                    st.session_state.notebook_id = new_nb["id"]
                    st.session_state.notebook_title = new_nb["title"]
                    st.session_state.view_mode = "studio"
                    st.session_state.show_create_modal = False
                    st.query_params["notebook_id"] = new_nb["id"]
                    st.rerun()
            with col_c2:
                if st.button("Cancel"):
                    st.session_state.show_create_modal = False
                    st.rerun()

    # Platform Telemetry Metrics
    all_notebooks = Storage.list_notebooks()
    total_sources = sum(len(Storage.get_sources(nb["id"])) for nb in all_notebooks)
    total_artifacts = sum(len(Storage.get_notebook_artifacts(nb["id"])) for nb in all_notebooks)
    matrix_stats = groq_router.get_router_matrix_stats()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cached Notebooks", len(all_notebooks))
    m2.metric("Ingested Lectures", total_sources)
    m3.metric("Synthesized Compendiums", total_artifacts)
    m4.metric("Active Groq Keys", f"{matrix_stats.get('total_keys', 8)}/8 Slots Active")

    st.divider()

    if not all_notebooks:
        st.info("👋 You don't have any notebooks yet. Click **➕ Create New Notebook** above to ingest YouTube videos and generate study notes.")
    else:
        # Search & Filter
        search_query = st.text_input("Search notebooks...", placeholder="Type to filter notebooks by title...", label_visibility="collapsed")
        filtered_notebooks = [nb for nb in all_notebooks if search_query.lower() in nb["title"].lower()] if search_query else all_notebooks

        # Grid of Notebook Cards
        for i in range(0, len(filtered_notebooks), 2):
            row_cols = st.columns(2)
            for j in range(2):
                if i + j < len(filtered_notebooks):
                    nb = filtered_notebooks[i + j]
                    sources = Storage.get_sources(nb["id"])
                    artifacts = Storage.get_notebook_artifacts(nb["id"])
                    ready_sources = sum(1 for s in sources if s.get("status") == "ready")
                    updated_dt = datetime.fromtimestamp(nb.get("updated_at", time.time())).strftime("%b %d, %Y • %I:%M %p")

                    with row_cols[j]:
                        with st.container(border=True):
                            st.markdown(f"### 📓 {nb['title']}")
                            st.caption(f"🕒 Last Active: {updated_dt}")
                            
                            c_info1, c_info2 = st.columns(2)
                            c_info1.markdown(f"🎬 **{len(sources)} Sources** ({ready_sources} Ready)")
                            c_info2.markdown(f"📚 **{len(artifacts)} Studio Notes**")

                            st.divider()
                            c_btn1, c_btn2, c_btn3 = st.columns([3, 1, 1])
                            with c_btn1:
                                if st.button(f"🚀 Open Notebook", key=f"open_{nb['id']}", type="primary"):
                                    st.session_state.notebook_id = nb["id"]
                                    st.session_state.notebook_title = nb["title"]
                                    st.session_state.view_mode = "studio"
                                    st.session_state.current_artifact = artifacts[0] if artifacts else None
                                    st.query_params["notebook_id"] = nb["id"]
                                    st.rerun()
                            with c_btn2:
                                if st.button("✏️", key=f"ren_card_{nb['id']}", help="Rename notebook"):
                                    st.session_state[f"rename_active_{nb['id']}"] = not st.session_state.get(f"rename_active_{nb['id']}", False)
                                    st.rerun()
                            with c_btn3:
                                if st.button("🗑️", key=f"del_card_{nb['id']}", help="Delete notebook"):
                                    Storage.delete_notebook(nb["id"])
                                    st.rerun()

                            if st.session_state.get(f"rename_active_{nb['id']}", False):
                                ren_text = st.text_input("New Name", value=nb["title"], key=f"input_ren_{nb['id']}")
                                if st.button("Save", key=f"save_ren_{nb['id']}"):
                                    if ren_text.strip():
                                        Storage.rename_notebook(nb["id"], ren_text.strip())
                                    st.session_state[f"rename_active_{nb['id']}"] = False
                                    st.rerun()

    st.stop()

# =========================================================================
# 🎓 2. STUDIO VIEW (Active Workspace)
# =========================================================================
active_nb_id = st.session_state.get("notebook_id") or param_nb_id
active_nb = Storage.get_notebook(active_nb_id) if active_nb_id else None

if not active_nb:
    st.session_state.view_mode = "gallery"
    st.session_state.notebook_id = None
    st.query_params.clear()
    st.rerun()

notebook_id = active_nb["id"]
st.session_state.notebook_id = notebook_id
st.session_state.notebook_title = active_nb["title"]
st.query_params["notebook_id"] = notebook_id

# Restore current studio artifact if not in session state
if "current_artifact" not in st.session_state or st.session_state.current_artifact is None:
    saved_artifacts = Storage.get_notebook_artifacts(notebook_id)
    st.session_state.current_artifact = saved_artifacts[0] if saved_artifacts else None

# ----------------- Sidebar: Sources & Notebook Manager -----------------
with st.sidebar:
    if st.button("⬅ 🏠 All Notebooks"):
        st.session_state.view_mode = "gallery"
        st.session_state.notebook_id = None
        st.query_params.clear()
        st.rerun()

    st.divider()
    st.title("🎓 YouTube NotebookLM")
    st.markdown(":green-background[⚡ 8-Key Groq Matrix Active]")
    st.divider()

    # Notebook Header / Rename
    col_nb1, col_nb2 = st.columns([4, 1])
    with col_nb1:
        st.subheader(f"📓 {st.session_state.get('notebook_title', 'Notebook')}")
    with col_nb2:
        if st.button("✏️", help="Rename this notebook"):
            st.session_state.is_renaming_nb = not st.session_state.get("is_renaming_nb", False)
            st.rerun()

    if st.session_state.get("is_renaming_nb", False):
        with st.container():
            new_nb_name = st.text_input(
                "Edit Notebook Name",
                value=st.session_state.get("notebook_title", "Notebook"),
                key="rename_nb_field"
            )
            col_save, col_cancel = st.columns(2)
            with col_save:
                if st.button("💾 Save"):
                    if new_nb_name.strip():
                        Storage.rename_notebook(notebook_id, new_nb_name.strip())
                        st.session_state.notebook_title = new_nb_name.strip()
                    st.session_state.is_renaming_nb = False
                    st.rerun()
            with col_cancel:
                if st.button("Cancel"):
                    st.session_state.is_renaming_nb = False
                    st.rerun()

    # Notebook Switcher Dropdown
    all_notebooks = Storage.list_notebooks()
    if len(all_notebooks) > 1:
        nb_options = {nb["id"]: nb["title"] for nb in all_notebooks}
        current_idx = list(nb_options.keys()).index(notebook_id) if notebook_id in nb_options else 0
        selected_nb_id = st.selectbox(
            "Switch Notebook",
            options=list(nb_options.keys()),
            format_func=lambda x: nb_options[x],
            index=current_idx,
            label_visibility="collapsed"
        )
        if selected_nb_id != notebook_id:
            st.session_state.notebook_id = selected_nb_id
            st.session_state.notebook_title = nb_options[selected_nb_id]
            st.session_state.chat_history = Storage.get_chat_history(selected_nb_id)
            artifacts = Storage.get_notebook_artifacts(selected_nb_id)
            st.session_state.current_artifact = artifacts[0] if artifacts else None
            st.session_state.is_renaming_nb = False
            st.query_params["notebook_id"] = selected_nb_id
            st.rerun()

    st.divider()
    st.subheader("📥 Ingest Video or Playlist")
    
    url_input = st.text_input(
        "YouTube URL",
        placeholder="Paste YouTube Video or Playlist URL...",
        label_visibility="collapsed"
    )
    
    def process_source_pipeline(src, current_notebook_id, status_placeholder=None):
        """Processes a single source: checks cache -> download -> whisper -> topic index -> cleanup."""
        vid_id = src.get("video_id", "") or YouTubeDownloader.extract_video_id(src.get("url", ""))

        # ⚡ 1. Zero-Download Cache Check
        cached_trans = Storage.get_transcript_by_video_id(vid_id)
        if cached_trans:
            if status_placeholder:
                status_placeholder.caption(f"⚡ Instant Cache Hit: {src.get('title', 'video')[:30]}...")
            logger.info(f"⚡ [Cache Hit] Reusing transcript for {vid_id} without downloading!")
            Storage.save_transcript(src["id"], cached_trans["full_text"], cached_trans["segments"])
            
            cached_topics = Storage.get_topic_index_by_video_id(vid_id)
            if cached_topics:
                for t in cached_topics:
                    t["source_id"] = src["id"]
                Storage.save_topic_index(src["id"], current_notebook_id, cached_topics)

            Storage.update_source_status(
                src["id"],
                status="ready",
                progress=100.0,
                title=cached_trans.get("title") or src.get("title"),
                duration=cached_trans.get("duration") or src.get("duration", 0),
                channel=cached_trans.get("channel") or src.get("channel", "YouTube")
            )
            return

        # 2. Audio Download
        if status_placeholder:
            status_placeholder.caption(f"📥 Downloading Audio: {src.get('title', 'video')[:30]}...")
        def update_dl_pct(p, msg):
            pass
        meta = YouTubeDownloader.download_audio(src["url"], src["id"], progress_callback=update_dl_pct)
        Storage.update_source_status(
            src["id"],
            status="transcribing",
            progress=55.0,
            title=meta["title"],
            video_id=meta["video_id"],
            channel=meta["channel"],
            duration=meta["duration"],
            thumbnail_url=meta["thumbnail_url"],
            audio_path=meta["audio_path"],
            chapters=meta["chapters"]
        )

        # 3. Whisper Large Transcription
        if status_placeholder:
            status_placeholder.caption(f"🎙️ Whisper Transcription: {meta.get('title', 'video')[:30]}...")
        Transcriber.process_source_audio(
            source_id=src["id"],
            audio_path=meta["audio_path"]
        )

        # 4. Topic Indexing
        if status_placeholder:
            status_placeholder.caption(f"🧠 Indexing Topics: {meta.get('title', 'video')[:30]}...")
        try:
            TopicIndexer.index_source_topics(src["id"])
        except Exception as te:
            logger.warning(f"Topic indexing warning: {te}")

        # 5. Audio Cleanup
        YouTubeDownloader.cleanup_audio_files(src["id"], meta.get("audio_path"))

        # 6. Mark Ready
        Storage.update_source_status(src["id"], status="ready", progress=100.0)

    if st.button("🚀 Ingest & Transcribe"):
        if url_input.strip():
            with st.spinner("Discovering playlist and video metadata..."):
                try:
                    videos = YouTubeDownloader.get_playlist_videos(url_input.strip())
                    st.info(f"Found {len(videos)} video(s). Ingesting...")
                    
                    registered = []
                    for v in videos:
                        src = Storage.add_source(
                            notebook_id=notebook_id,
                            url=v["url"],
                            title=v.get("title", "Loading video..."),
                            video_id=v.get("video_id", ""),
                            channel=v.get("channel", ""),
                            duration=v.get("duration", 0),
                            thumbnail_url=v.get("thumbnail_url", "")
                        )
                        registered.append(src)

                    prog_box = st.empty()
                    prog_bar = st.progress(0)
                    for idx, src in enumerate(registered):
                        prog_bar.progress(int((idx / len(registered)) * 100))
                        process_source_pipeline(src, notebook_id, status_placeholder=prog_box)

                    prog_bar.progress(100)
                    prog_box.success("✅ Ingestion Complete!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Ingestion Error: {ex}")
        else:
            st.warning("Please enter a valid YouTube URL.")

    st.divider()
    
    # Sources List
    sources = Storage.get_sources(notebook_id)
    ready_count = len([s for s in sources if s.get("status") == "ready"])
    queued_sources = [s for s in sources if s.get("status") != "ready"]
    
    st.subheader(f"📚 Sources ({ready_count}/{len(sources)} Ready)")

    if queued_sources:
        if st.button(f"▶ Resume Ingestion ({len(queued_sources)} Queued)", type="primary"):
            status_box = st.empty()
            prog_bar = st.progress(0)
            for idx, q_src in enumerate(queued_sources):
                pct = int(((idx) / len(queued_sources)) * 100)
                prog_bar.progress(pct)
                try:
                    process_source_pipeline(q_src, notebook_id, status_placeholder=status_box)
                except Exception as q_err:
                    logger.error(f"Error processing {q_src['id']}: {q_err}")
                    Storage.update_source_status(q_src["id"], status="error", error_message=str(q_err))

            prog_bar.progress(100)
            status_box.success("✅ Ingestion Complete!")
            st.rerun()
    
    if not sources:
        st.caption("No sources yet. Paste a YouTube URL above to begin.")
    else:
        for s in sources:
            thumb = s.get("thumbnail_url") or "https://images.unsplash.com/photo-1611162617213-7d7a39e9b1d7?w=160&q=80"
            dur_min = int(s.get("duration", 0) // 60)
            dur_sec = int(s.get("duration", 0) % 60)
            dur_str = f"{dur_min}m {dur_sec}s" if s.get("duration") else ""
            status = s.get("status", "pending")
            
            with st.expander(f"🎬 {s.get('title', 'Video')[:28]}...", expanded=False):
                st.image(thumb)
                st.caption(f"Channel: {s.get('channel', 'YouTube')} | Duration: {dur_str}")
                if status == "ready":
                    st.markdown(":green[✓ Ready]")
                elif status == "transcribing":
                    st.markdown(":blue[⏳ Transcribing...]")
                elif status == "error":
                    st.markdown(":red[⚠️ Error]")
                else:
                    st.caption("Queued")

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    if status != "ready":
                        if st.button("▶ Ingest", key=f"retry_{s['id']}"):
                            with st.spinner("Processing video..."):
                                process_source_pipeline(s, notebook_id)
                            st.rerun()
                with col_s2:
                    if st.button("🗑️ Delete", key=f"del_{s['id']}"):
                        Storage.delete_source(s["id"])
                        st.rerun()

    # 8-Key Live Monitor
    st.divider()
    with st.expander("⚡ 8-Key Groq Matrix Monitor", expanded=False):
        matrix_stats = groq_router.get_router_matrix_stats()
        st.caption(f"Active Slots: {matrix_stats.get('total_keys', 8)} Keys • 7 Models")
        for k in matrix_stats.get("key_stats", []):
            st.markdown(f"**Key #{k['key_index']}** `{k['masked_key']}`: 🎙️ {k['transcriptions']} | 💬 {k['completions']}")

# ----------------- Main Workspace Tabs -----------------
st.title(f"📖 {st.session_state.get('notebook_title', 'Research Studio')}")
st.caption("Grounded Multi-Modal Research Assistant • LaTeX Compendiums • PDF Publishing")

tab_chat, tab_studio, tab_transcripts = st.tabs([
    "💬 AI Chat & Grounded RAG",
    "📚 Studio Notes & Exporters",
    "🎙️ Transcripts & Topic Index"
])

# ----------------- TAB 1: Grounded RAG Chat -----------------
with tab_chat:
    chat_history = Storage.get_chat_history(notebook_id)
    
    if not chat_history:
        st.info("💡 **Welcome to your Grounded Workspace!** Ask any question across your ingested YouTube lectures. Every answer is substantiated with exact timestamp citations.")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            if st.button("📌 Executive Summary & Core Theses"):
                st.session_state.pending_prompt = "Provide a comprehensive executive summary with key takeaways and principles from all ingested videos."
                st.rerun()
        with col_p2:
            if st.button("💻 Extract Theorems, Formulas & Algorithms"):
                st.session_state.pending_prompt = "Extract all mathematical formulas, core algorithms, and technical concepts mentioned across the lectures."
                st.rerun()
        with col_p3:
            if st.button("❓ Active Recall Practice Quiz"):
                st.session_state.pending_prompt = "Generate a 5-question active recall test with solutions based on these video topics."
                st.rerun()
    else:
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            citations = msg.get("citations", [])
            with st.chat_message(role):
                st.markdown(content)
                if citations:
                    with st.expander(f"📌 {len(citations)} Source Citation(s)", expanded=False):
                        for c in citations:
                            st.markdown(f"**{c.get('source_title', 'Video')}** `[{c.get('timestamp_str', '00:00')}]`\n> {c.get('text', '')}")

    # Chat Input
    prompt_val = st.session_state.pop("pending_prompt", None)
    user_query = st.chat_input("Ask a question about your videos (e.g., 'Explain the deprivation account of death with timestamps')...") or prompt_val

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        Storage.add_chat_message(notebook_id, role="user", content=user_query)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                stream_gen, citations = NotebookAgent.answer_rag_stream(notebook_id, user_query)
                for chunk in stream_gen:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                
                if citations:
                    with st.expander(f"📌 {len(citations)} Source Citation(s)", expanded=True):
                        for c in citations:
                            st.markdown(f"**{c.get('source_title', 'Video')}** `[{c.get('timestamp_str', '00:00')}]`\n> {c.get('text', '')}")

                Storage.add_chat_message(notebook_id, role="assistant", content=full_response, citations=citations)
            except Exception as e:
                st.error(f"Error answering query: {e}")

# ----------------- TAB 2: Studio Notes & Multi-Format Exporters -----------------
with tab_studio:
    st.subheader("🛠️ Study Studio & Multi-Format Exporters")
    st.caption("Generate publication-ready academic documents and download as **Compiled PDF**, **Academic LaTeX**, **Markdown**, or **Standalone HTML**.")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    with col_btn1:
        if st.button("⚡ Master Course Book (.PDF)", help="Multi-key parallel map-reduce across all playlist lectures"):
            with st.spinner("Synthesizing Master Course Textbook across parallel Groq keys..."):
                try:
                    art = ParallelSynthesizer.synthesize_master_booklet(notebook_id)
                    st.session_state.current_artifact = art
                    st.success("Master Course Booklet synthesized!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Synthesis failed: {e}")
    with col_btn2:
        sources_ready = [s for s in Storage.get_sources(notebook_id) if s.get("status") == "ready"]
        if sources_ready:
            selected_source_for_note = st.selectbox("Select Lecture", options=sources_ready, format_func=lambda x: x["title"], label_visibility="collapsed")
            if st.button("📝 Single Lecture Compendium"):
                with st.spinner(f"Generating Comprehensive Lecture Notes for {selected_source_for_note['title'][:25]}..."):
                    try:
                        art = ParallelSynthesizer.synthesize_single_lecture(notebook_id, selected_source_for_note["id"])
                        st.session_state.current_artifact = art
                        st.success("Lecture Note generated!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lecture note generation failed: {e}")
        else:
            st.button("📝 Single Lecture Compendium", disabled=True)
    with col_btn3:
        if st.button("🧠 Study Guide & Active Recall"):
            with st.spinner("Generating active recall study guide..."):
                try:
                    art = NotebookAgent.generate_study_guide(notebook_id)
                    st.session_state.current_artifact = art
                    st.success("Study Guide generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Study guide generation failed: {e}")
    with col_btn4:
        if st.button("🗺️ Concept Mind Map"):
            with st.spinner("Building concept architecture..."):
                try:
                    art = NotebookAgent.generate_mindmap(notebook_id)
                    st.session_state.current_artifact = art
                    st.success("Mind Map generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Mind map generation failed: {e}")

    st.divider()

    # Render Current Artifact & History Switcher
    artifacts = Storage.get_notebook_artifacts(notebook_id)
    if artifacts:
        # Artifact History Selector Dropdown
        col_art_select, col_art_meta = st.columns([3, 1])
        with col_art_select:
            art_dict = {a["id"]: a for a in artifacts}
            current_art_id = st.session_state.get("current_artifact", {}).get("id") if st.session_state.get("current_artifact") else artifacts[0]["id"]
            selected_art_id = st.selectbox(
                "Select Generated Document",
                options=list(art_dict.keys()),
                format_func=lambda x: f"📄 {art_dict[x].get('title', 'Note')} ({datetime.fromtimestamp(art_dict[x].get('created_at', time.time())).strftime('%b %d • %I:%M %p')})",
                index=list(art_dict.keys()).index(current_art_id) if current_art_id in art_dict else 0
            )
            current_art = art_dict[selected_art_id]
            st.session_state.current_artifact = current_art

        with col_art_meta:
            created_dt = datetime.fromtimestamp(current_art.get("created_at", time.time())).strftime("%b %d, %I:%M %p")
            st.caption(f"Created: {created_dt}")

        st.subheader(f"📖 {current_art.get('title', 'Study Document')}")
        
        # 4 Export Download Buttons
        doc_title = current_art.get("title", "Study_Notes")
        md_content = current_art.get("content_md", "")

        exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
        
        # 1. PDF
        with exp_col1:
            try:
                pdf_filename = Exporter.generate_pdf(doc_title, md_content)
                pdf_path = EXPORTS_DIR / pdf_filename
                if pdf_path.exists():
                    with open(pdf_path, "rb") as pf:
                        st.download_button(
                            label="📕 Download .PDF",
                            data=pf.read(),
                            file_name=pdf_filename,
                            mime="application/pdf"
                        )
            except Exception as pe:
                st.caption(f"PDF generation: {pe}")

        # 2. Academic LaTeX
        with exp_col2:
            latex_content = Exporter.generate_latex(doc_title, md_content)
            st.download_button(
                label="📐 Download .TEX (LaTeX)",
                data=latex_content,
                file_name=f"{doc_title}.tex",
                mime="text/x-tex"
            )

        # 3. Markdown
        with exp_col3:
            st.download_button(
                label="📄 Download .MD",
                data=md_content,
                file_name=f"{doc_title}.md",
                mime="text/markdown"
            )

        # 4. Standalone HTML
        with exp_col4:
            html_content = Exporter.generate_html(doc_title, md_content)
            st.download_button(
                label="🌐 Download .HTML",
                data=html_content,
                file_name=f"{doc_title}.html",
                mime="text/html"
            )

        st.divider()
        st.markdown(md_content)
    else:
        st.info("No study artifacts generated yet. Click any button above to synthesize your videos!")

# ----------------- TAB 3: Transcripts & Topics Index -----------------
with tab_transcripts:
    st.subheader("🎙️ Lecture Transcripts & Indexed Topic Boundaries")
    
    sources_with_transcripts = [s for s in Storage.get_sources(notebook_id) if s.get("status") == "ready"]
    if not sources_with_transcripts:
        st.info("Ingest and transcribe YouTube videos to inspect transcripts and extracted topic indices.")
    else:
        selected_source = st.selectbox(
            "Select Lecture Transcript",
            options=sources_with_transcripts,
            format_func=lambda x: f"{x['title']} ({int(x.get('duration', 0)//60)} mins)"
        )
        
        if selected_source:
            tab_top, tab_raw = st.tabs(["🗺️ Extracted Topic Architecture", "📜 Full Timestamped Transcript"])
            
            with tab_top:
                topics = Storage.get_source_topic_index(selected_source["id"])
                if topics:
                    for t in topics:
                        with st.expander(f"📍 {t.get('title', 'Topic')} `[{t.get('start_time_str', '00:00')} - {t.get('end_time_str', '00:00')}]`", expanded=False):
                            st.markdown(f"**Summary:** {t.get('summary', '')}")
                            if t.get("key_takeaway"):
                                st.markdown(f"**Key Takeaway:** {t.get('key_takeaway')}")
                            if t.get("keywords"):
                                st.caption(f"Keywords: {', '.join(t.get('keywords', []))}")
                else:
                    st.caption("Topic indexing in progress or empty for this source.")

            with tab_raw:
                trans = Storage.get_transcript(selected_source["id"])
                if trans and trans.get("segments"):
                    t_filter = st.text_input("Filter transcript lines...", placeholder="Search keywords across timestamps...", label_visibility="collapsed")
                    filtered_segs = [s for s in trans["segments"] if t_filter.lower() in s.get("text", "").lower()] if t_filter else trans["segments"]
                    
                    with st.container(height=500):
                        for seg in filtered_segs:
                            st.markdown(f"**`[{seg.get('timestamp_str', '00:00')}]`** {seg.get('text', '')}")
                else:
                    st.caption("No transcript segments found.")
