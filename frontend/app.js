// Global State
let currentNotebookId = null;
let currentArtifact = null;
let activeSources = [];
let pollInterval = null;

// Initialize App
document.addEventListener("DOMContentLoaded", () => {
  initMermaid();
  loadDefaultNotebook();
  setupEventListeners();
  startKeyStatsPolling();
});

function initMermaid() {
  if (window.mermaid) {
    mermaid.initialize({
      startOnLoad: false,
      theme: 'dark',
      themeVariables: {
        darkMode: true,
        background: '#0b0f19',
        primaryColor: '#6366f1',
        primaryTextColor: '#fff',
        primaryBorderColor: '#818cf8',
        lineColor: '#38bdf8',
        secondaryColor: '#1e293b',
        tertiaryColor: '#0f172a'
      }
    });
  }
}

// ----------------- Notebook Management -----------------
async function loadDefaultNotebook() {
  try {
    const res = await fetch("/api/notebooks");
    const notebooks = await res.json();
    if (notebooks.length > 0) {
      currentNotebookId = notebooks[0].id;
      document.getElementById("currentNotebookTitle").textContent = notebooks[0].title;
    } else {
      const createRes = await fetch("/api/notebooks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "YouTube Master Class Notebook" })
      });
      const newNb = await createRes.json();
      currentNotebookId = newNb.id;
      document.getElementById("currentNotebookTitle").textContent = newNb.title;
    }
    loadNotebookDetails(currentNotebookId);
  } catch (err) {
    console.error("Failed loading default notebook:", err);
  }
}

async function loadNotebookDetails(notebookId) {
  try {
    const res = await fetch(`/api/notebooks/${notebookId}`);
    if (!res.ok) return;
    const data = await res.json();

    // Render Sources
    renderSources(data.sources);

    // Render Chat History
    renderChatHistory(data.chat_history);

    // Render Latest Artifact if available
    if (data.artifacts && data.artifacts.length > 0) {
      displayArtifact(data.artifacts[0]);
    }

    // Start progress polling if any source is pending/downloading/transcribing
    checkActiveSourcePolling(data.sources);
  } catch (err) {
    console.error("Error loading notebook details:", err);
  }
}

// ----------------- Sources Ingestion & Rendering -----------------
function renderSources(sources) {
  activeSources = sources || [];
  const listEl = document.getElementById("sourcesList");
  const countEl = document.getElementById("sourcesCount");
  const activeCountBadge = document.getElementById("activeSourcesCountBadge");
  const sourceSelect = document.getElementById("transcriptSourceSelect");

  countEl.textContent = activeSources.length;
  const readyCount = activeSources.filter(s => s.status === 'ready').length;
  activeCountBadge.textContent = `${readyCount} / ${activeSources.length} sources ready`;

  // Update transcript dropdown
  sourceSelect.innerHTML = '<option value="">Select video source...</option>';
  activeSources.forEach(s => {
    if (s.status === 'ready') {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = `${s.title} (${Math.round(s.duration)}s)`;
      sourceSelect.appendChild(opt);
    }
  });

  if (activeSources.length === 0) {
    listEl.innerHTML = `
      <div class="empty-state">
        <i class="fa-brands fa-youtube empty-icon"></i>
        <p>No video sources added yet</p>
        <span>Paste a YouTube video or playlist link above to transcribe with Groq Whisper Large</span>
      </div>
    `;
    return;
  }

  listEl.innerHTML = "";
  activeSources.forEach(source => {
    const card = document.createElement("div");
    card.className = "source-card";
    card.id = `source-card-${source.id}`;

    let statusHtml = "";
    if (source.status === "ready") {
      statusHtml = `<span class="status-badge status-ready"><i class="fa-solid fa-circle-check"></i> Ready</span>`;
    } else if (source.status === "transcribing") {
      statusHtml = `<span class="status-badge status-transcribing"><i class="fa-solid fa-spinner fa-spin"></i> Transcribing (${Math.round(source.progress || 60)}%)</span>`;
    } else if (source.status === "downloading") {
      statusHtml = `<span class="status-badge status-downloading"><i class="fa-solid fa-cloud-arrow-down fa-bounce"></i> Downloading (${Math.round(source.progress || 25)}%)</span>`;
    } else if (source.status === "error") {
      statusHtml = `<span class="status-badge status-error" title="${source.error_message || ''}"><i class="fa-solid fa-triangle-exclamation"></i> Error</span>`;
    } else {
      statusHtml = `<span class="status-badge"><i class="fa-regular fa-clock"></i> Queued</span>`;
    }

    const durationStr = source.duration ? `${Math.floor(source.duration / 60)}m ${Math.floor(source.duration % 60)}s` : "";
    const thumbUrl = source.thumbnail_url || "https://images.unsplash.com/photo-1611162617213-7d7a39e9b1d7?w=160&q=80";

    card.innerHTML = `
      <div class="source-top">
        <img src="${thumbUrl}" class="source-thumb" alt="Thumbnail" onerror="this.src='https://images.unsplash.com/photo-1611162617213-7d7a39e9b1d7?w=160&q=80'">
        <div class="source-meta">
          <div class="source-title" title="${source.title}">${source.title}</div>
          <div class="source-sub">
            <span>${source.channel || 'YouTube'}</span>
            <span>${durationStr}</span>
          </div>
        </div>
      </div>
      <div class="source-bottom">
        <div class="d-flex justify-content-between align-items-center" style="display:flex; justify-content:space-between; align-items:center;">
          ${statusHtml}
          <div class="source-actions">
            ${source.status === 'ready' ? `
              <button class="btn-xs btn-primary-subtle" onclick="generateLectureNote('${source.id}')" title="Generate Lecture Note"><i class="fa-solid fa-pen-nib"></i> Note</button>
              <button class="icon-btn-subtle" onclick="viewSourceTranscript('${source.id}')" title="View Transcript"><i class="fa-solid fa-file-lines"></i></button>
            ` : ''}
            <button class="icon-btn-subtle" onclick="deleteSource('${source.id}')" title="Remove Source"><i class="fa-solid fa-trash-can"></i></button>
          </div>
        </div>
        ${(source.status === 'downloading' || source.status === 'transcribing') ? `
          <div class="progress-bar-container" style="margin-top:6px;">
            <div class="progress-bar-fill" style="width: ${source.progress || 20}%;"></div>
          </div>
        ` : ''}
      </div>
    `;

    listEl.appendChild(card);
  });
}

function checkActiveSourcePolling(sources) {
  const hasInProgress = sources.some(s => s.status === 'pending' || s.status === 'downloading' || s.status === 'transcribing');
  if (hasInProgress) {
    if (!pollInterval) {
      pollInterval = setInterval(() => {
        loadNotebookDetails(currentNotebookId);
      }, 3000);
    }
  } else {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  }
}

async function ingestUrl(url) {
  if (!url || !url.trim()) return;
  try {
    const res = await fetch("/api/sources/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notebook_id: currentNotebookId, url: url.trim() })
    });
    if (res.ok) {
      // Reload notebook details to see newly created source
      setTimeout(() => loadNotebookDetails(currentNotebookId), 800);
    }
  } catch (err) {
    console.error("Ingestion failed:", err);
  }
}

async function deleteSource(sourceId) {
  if (!confirm("Are you sure you want to remove this source?")) return;
  try {
    await fetch(`/api/sources/${sourceId}`, { method: "DELETE" });
    loadNotebookDetails(currentNotebookId);
  } catch (err) {
    console.error("Delete source failed:", err);
  }
}

// ----------------- Transcript & Audio Sync -----------------
async function viewSourceTranscript(sourceId) {
  switchStudioTab("transcript");
  const sourceSelect = document.getElementById("transcriptSourceSelect");
  sourceSelect.value = sourceId;
  loadTranscriptView(sourceId);
}

async function loadTranscriptView(sourceId) {
  if (!sourceId) return;
  const container = document.getElementById("transcriptScrollArea");
  const audioWrapper = document.getElementById("audioPlayerWrapper");
  const audioPlayer = document.getElementById("inAppAudioPlayer");

  container.innerHTML = '<div class="p-4 text-center"><i class="fa-solid fa-spinner fa-spin"></i> Loading transcript...</div>';

  try {
    const res = await fetch(`/api/sources/${sourceId}`);
    const data = await res.json();
    const source = data.source;
    const transcript = data.transcript;

    if (source.audio_path) {
      const audioFilename = source.audio_path.split("/").pop();
      audioPlayer.src = `/api/audio/${audioFilename}`;
      audioWrapper.style.display = "block";
    } else {
      audioWrapper.style.display = "none";
    }

    if (!transcript || !transcript.segments || transcript.segments.length === 0) {
      container.innerHTML = `<div class="empty-state"><p>No transcript segments available.</p></div>`;
      return;
    }

    container.innerHTML = "";
    transcript.segments.forEach(seg => {
      const segEl = document.createElement("div");
      segEl.className = "transcript-segment-card";
      segEl.dataset.start = seg.start;
      segEl.innerHTML = `
        <span class="seg-timestamp" onclick="seekAudioPlayer(${seg.start})">${seg.timestamp_str}</span>
        <span class="seg-text">${seg.text}</span>
      `;
      container.appendChild(segEl);
    });
  } catch (err) {
    container.innerHTML = `<p class="text-danger p-4">Failed to load transcript.</p>`;
  }
}

function seekAudioPlayer(seconds) {
  const player = document.getElementById("inAppAudioPlayer");
  if (player && player.src) {
    player.currentTime = seconds;
    player.play();
  }
}

// ----------------- Agentic RAG Chat -----------------
function renderChatHistory(messages) {
  const container = document.getElementById("chatConversation");
  if (!messages || messages.length === 0) return;

  container.innerHTML = "";
  messages.forEach(msg => {
    appendMessageToChat(msg.role, msg.content, msg.citations);
  });
}

function appendMessageToChat(role, text, citations = []) {
  const container = document.getElementById("chatConversation");
  
  // Remove welcome card if first message
  const welcomeCard = container.querySelector(".chat-welcome-card");
  if (welcomeCard) welcomeCard.remove();

  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.innerHTML = role === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-sparkles"></i>';

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  
  // Parse citations into interactive tags
  let formattedText = text;
  if (citations && citations.length > 0) {
    citations.forEach(c => {
      const citationRegex = new RegExp(`\\[${c.citation_id.replace(/[\[\]]/g, '')}\\]`, 'g');
      const tagHtml = `<span class="citation-tag" onclick="jumpToCitation('${c.source_id}', ${c.start_seconds})" title="${c.source_title} at ${c.timestamp_str}"><i class="fa-solid fa-video"></i> ${c.timestamp_str}</span>`;
      formattedText = formattedText.replace(citationRegex, tagHtml);
    });
  }

  bubble.innerHTML = marked.parse(formattedText);

  row.appendChild(avatar);
  row.appendChild(bubble);
  container.appendChild(row);

  container.scrollTop = container.scrollHeight;
  return bubble;
}

function jumpToCitation(sourceId, seconds) {
  viewSourceTranscript(sourceId);
  setTimeout(() => {
    seekAudioPlayer(seconds);
  }, 400);
}

async function sendChatMessage(queryText) {
  const inputEl = document.getElementById("chatInput");
  const query = queryText || inputEl.value.trim();
  if (!query) return;

  inputEl.value = "";
  appendMessageToChat("user", query);

  // Create assistant placeholder bubble
  const assistantBubble = appendMessageToChat("assistant", '<i class="fa-solid fa-spinner fa-spin"></i> Synthesizing from video knowledge base...');

  try {
    const sseUrl = `/api/chat/stream?notebook_id=${currentNotebookId}&query=${encodeURIComponent(query)}`;
    const eventSource = new EventSource(sseUrl);

    let accumulatedText = "";
    let citations = [];

    eventSource.addEventListener("citations", (e) => {
      citations = JSON.parse(e.data);
    });

    eventSource.addEventListener("token", (e) => {
      const data = JSON.parse(e.data);
      accumulatedText += data.token;
      
      let parsed = accumulatedText;
      if (citations && citations.length > 0) {
        citations.forEach(c => {
          const citationRegex = new RegExp(`\\[${c.citation_id.replace(/[\[\]]/g, '')}\\]`, 'g');
          const tagHtml = `<span class="citation-tag" onclick="jumpToCitation('${c.source_id}', ${c.start_seconds})" title="${c.source_title} at ${c.timestamp_str}"><i class="fa-solid fa-video"></i> ${c.timestamp_str}</span>`;
          parsed = parsed.replace(citationRegex, tagHtml);
        });
      }
      assistantBubble.innerHTML = marked.parse(parsed);
      document.getElementById("chatConversation").scrollTop = document.getElementById("chatConversation").scrollHeight;
    });

    eventSource.addEventListener("done", () => {
      eventSource.close();
    });

    eventSource.onerror = (err) => {
      console.error("SSE Error:", err);
      eventSource.close();
    };

  } catch (err) {
    assistantBubble.innerHTML = `<p class="text-danger">Failed to generate answer.</p>`;
  }
}

window.sendSuggestedQuery = function(query) {
  sendChatMessage(query);
};

// ----------------- Studio Artifacts & Exports -----------------
async function generateLectureNote(sourceId) {
  const container = document.getElementById("notesMarkdownView");
  switchStudioTab("notes");
  
  const src = activeSources.find(s => s.id === sourceId);
  const srcTitle = src ? src.title : "Lecture";

  container.innerHTML = `
    <div class="empty-studio-state">
      <i class="fa-solid fa-pen-nib fa-spin"></i>
      <h4>Writing Lecture Notes for "${srcTitle}"...</h4>
      <p>Extracting formulas, code, timestamps, and deep conceptual breakdowns using Groq 8-Key Engine.</p>
    </div>
  `;

  try {
    const res = await fetch("/api/artifacts/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notebook_id: currentNotebookId, type: "comprehensive_notes", source_id: sourceId })
    });
    
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Generation failed");
    }

    const artifact = await res.json();
    displayArtifact(artifact);
  } catch (err) {
    container.innerHTML = `<div class="empty-studio-state text-danger"><h4>Generation Error</h4><p>${err.message}</p></div>`;
  }
}

async function generateStudioArtifact(type, sourceId = null) {
  if (type === "single_lecture_dropdown") {
    const readySources = activeSources.filter(s => s.status === "ready");
    if (readySources.length === 0) {
      alert("Please wait until at least one video source finishes transcribing.");
      return;
    }
    if (readySources.length === 1) {
      return generateLectureNote(readySources[0].id);
    }
    // Prompt user to pick lecture
    const lectureTitles = readySources.map((s, i) => `${i + 1}. ${s.title}`).join("\n");
    const choice = prompt(`Select a lecture number to generate detailed notes:\n\n${lectureTitles}\n\nEnter number (1-${readySources.length}):`);
    const idx = parseInt(choice) - 1;
    if (!isNaN(idx) && readySources[idx]) {
      return generateLectureNote(readySources[idx].id);
    }
    return;
  }

  const container = document.getElementById("notesMarkdownView");
  switchStudioTab("notes");
  
  const loadingTitle = type === "master_booklet" 
    ? "Compiling Master Course Textbook (.PDF)..." 
    : "Synthesizing Studio Artifact...";

  const loadingDesc = type === "master_booklet"
    ? "Synthesizing all lectures in parallel across 8 Groq keys into a unified master book with Table of Contents, syllabus map, and PDF export."
    : "Consulting ingested transcripts and extracting deep structure with Groq LLaMA 3.3 70B & DeepSeek R1.";

  container.innerHTML = `
    <div class="empty-studio-state">
      <i class="fa-solid fa-wand-magic-sparkles fa-spin"></i>
      <h4>${loadingTitle}</h4>
      <p>${loadingDesc}</p>
    </div>
  `;

  try {
    const res = await fetch("/api/artifacts/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notebook_id: currentNotebookId, type: type === "master_booklet" ? "comprehensive_notes" : type, source_id: sourceId })
    });
    
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Generation failed");
    }

    const artifact = await res.json();
    displayArtifact(artifact);
  } catch (err) {
    container.innerHTML = `<div class="empty-studio-state text-danger"><h4>Generation Error</h4><p>${err.message}</p></div>`;
  }
}

function displayArtifact(art) {
  currentArtifact = art;
  document.getElementById("artifactTitle").textContent = art.title;
  document.getElementById("artifactTypeTag").textContent = art.type.replace("_", " ").toUpperCase();

  const notesView = document.getElementById("notesMarkdownView");
  notesView.innerHTML = marked.parse(art.content_md);

  // If mindmap, render in Mindmap tab as well
  if (art.type === "mindmap" || (art.metadata && art.metadata.mermaid)) {
    const mermaidCode = art.metadata?.mermaid || art.content_md;
    const cleanMermaid = extractMermaidCode(mermaidCode);
    renderMermaidDiagram(cleanMermaid);
  }

  // Setup export download buttons
  setupExportButtons(art);
}

function extractMermaidCode(text) {
  const match = text.match(/```mermaid([\s\S]*?)```/);
  return match ? match[1].trim() : text.trim();
}

function renderMermaidDiagram(code) {
  const container = document.getElementById("mindmapContainer");
  container.innerHTML = `<div class="mermaid">${code}</div>`;
  if (window.mermaid) {
    mermaid.init(undefined, container.querySelectorAll(".mermaid"));
  }
}

function setupExportButtons(art) {
  const mdBtn = document.getElementById("downloadMdBtn");
  const texBtn = document.getElementById("downloadTexBtn");
  const pdfBtn = document.getElementById("downloadPdfBtn");
  const htmlBtn = document.getElementById("downloadHtmlBtn");

  const safeTitle = sanitizeFilename(art.title);

  // 1. Markdown (.md)
  mdBtn.onclick = () => downloadBlob(art.content_md, `${safeTitle}.md`, "text/markdown");
  
  // 2. Academic LaTeX (.tex)
  if (art.content_tex) {
    texBtn.style.display = "inline-flex";
    texBtn.onclick = () => downloadBlob(art.content_tex, `${safeTitle}.tex`, "text/plain");
  } else {
    texBtn.onclick = () => {
      const tex = markdownToLatexClientFallback(art.title, art.content_md);
      downloadBlob(tex, `${safeTitle}.tex`, "text/plain");
    };
    texBtn.style.display = "inline-flex";
  }

  // 3. Compiled PDF (.pdf)
  if (art.pdf_path) {
    pdfBtn.style.display = "inline-flex";
    const pdfFilename = art.pdf_path.split("/").pop();
    pdfBtn.onclick = () => window.open(`/api/exports/${pdfFilename}`, "_blank");
  } else {
    pdfBtn.style.display = "inline-flex";
    pdfBtn.onclick = () => window.print();
  }

  // 4. Standalone Styled Web Notes (.html)
  htmlBtn.style.display = "inline-flex";
  htmlBtn.onclick = () => {
    const renderedBody = marked.parse(art.content_md);
    const standaloneHtml = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>${art.title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; background: #0b0f19; color: #f8fafc; line-height: 1.7; padding: 40px 20px; }
    .container { max-width: 860px; margin: auto; background: #111827; padding: 40px; border-radius: 16px; border: 1px solid #334155; }
    h1, h2, h3 { font-family: 'Outfit', sans-serif; color: #fff; }
    h1 { font-size: 26px; border-bottom: 2px solid #6366f1; padding-bottom: 10px; color: #818cf8; }
    h2 { font-size: 20px; color: #93c5fd; border-bottom: 1px solid #334155; padding-bottom: 6px; }
    p { margin: 14px 0; color: #e2e8f0; }
    blockquote { background: rgba(99, 102, 241, 0.1); border-left: 4px solid #6366f1; padding: 12px 18px; border-radius: 0 8px 8px 0; font-style: italic; }
    pre { background: #090d16; border: 1px solid #334155; padding: 16px; border-radius: 10px; overflow-x: auto; font-family: 'JetBrains Mono', monospace; font-size: 13px; }
    code { font-family: 'JetBrains Mono', monospace; color: #38bdf8; }
    .meta { font-size: 12px; color: #94a3b8; margin-bottom: 20px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="meta">YouTube NotebookLM &bull; Exported Knowledge Document</div>
    ${renderedBody}
  </div>
</body>
</html>`;
    downloadBlob(standaloneHtml, `${safeTitle}.html`, "text/html");
  };
}

function markdownToLatexClientFallback(title, md) {
  return `\\documentclass{article}
\\usepackage{amsmath,amssymb,hyperref,tcolorbox}
\\title{${title}}
\\date{\\today}
\\begin{document}
\\maketitle
${md.replace(/#/g, '%')}
\\end{document}`;
}

function downloadBlob(content, filename, contentType) {
  const blob = new Blob([content], { type: contentType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function sanitizeFilename(name) {
  return name.replace(/[^a-zA-Z0-9_-]/g, "_").substring(0, 40);
}

function switchStudioTab(tabName) {
  document.querySelectorAll(".studio-tab").forEach(tab => {
    tab.classList.toggle("active", tab.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-content").forEach(content => {
    content.classList.toggle("active", content.id === `${tabName}TabContent`);
  });
}

// ----------------- 8-Key Groq Health Monitor -----------------
async function startKeyStatsPolling() {
  async function fetchStats() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      
      const badgeCount = document.getElementById("groqKeyCount");
      if (badgeCount) badgeCount.textContent = `${data.total_keys} Groq Keys Active`;

      const grid = document.getElementById("keysGrid");
      if (grid && data.key_stats) {
        grid.innerHTML = "";
        data.key_stats.forEach(k => {
          const isAct = k.status === "Active";
          const card = document.createElement("div");
          card.className = "key-card";
          card.innerHTML = `
            <div class="key-card-top">
              <span class="key-title">Groq Key #${k.key_index}</span>
              <span class="key-status-dot ${isAct ? 'active' : 'cooldown'}">${k.status}</span>
            </div>
            <div class="text-muted" style="font-family: monospace; font-size: 11px;">${k.masked_key}</div>
            <div class="key-stats-row">
              <span>🎙️ Whisper: <b>${k.transcriptions}</b></span>
              <span>💬 LLM: <b>${k.completions}</b></span>
              <span>⚠️ Errors: <b>${k.errors}</b></span>
            </div>
          `;
          grid.appendChild(card);
        });
      }
    } catch (e) {
      console.warn("Failed fetching key stats:", e);
    }
  }

  fetchStats();
  setInterval(fetchStats, 5000);
}

// ----------------- Event Listeners Setup -----------------
function setupEventListeners() {
  // Quick Add
  const quickInput = document.getElementById("quickUrlInput");
  const quickBtn = document.getElementById("quickAddBtn");
  quickBtn.addEventListener("click", () => {
    if (quickInput.value.trim()) {
      ingestUrl(quickInput.value.trim());
      quickInput.value = "";
    }
  });
  quickInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      quickBtn.click();
    }
  });

  // Modal
  const modal = document.getElementById("addSourceModal");
  document.getElementById("openAddSourceModalBtn").addEventListener("click", () => {
    modal.style.display = "flex";
  });
  document.getElementById("closeSourceModalBtn").addEventListener("click", () => {
    modal.style.display = "none";
  });
  document.getElementById("cancelModalBtn").addEventListener("click", () => {
    modal.style.display = "none";
  });
  document.getElementById("submitIngestModalBtn").addEventListener("click", () => {
    const url = document.getElementById("modalUrlInput").value.trim();
    if (url) {
      ingestUrl(url);
      document.getElementById("modalUrlInput").value = "";
      modal.style.display = "none";
    }
  });

  // Chat Input
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendChatBtn");
  sendBtn.addEventListener("click", () => sendChatMessage());
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });

  // Studio Quick Action Chips
  document.querySelectorAll(".action-chip").forEach(btn => {
    btn.addEventListener("click", () => {
      const action = btn.dataset.action;
      generateStudioArtifact(action);
    });
  });

  // Studio Tabs
  document.querySelectorAll(".studio-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      switchStudioTab(tab.dataset.tab);
    });
  });

  // Transcript Dropdown
  document.getElementById("transcriptSourceSelect").addEventListener("change", (e) => {
    loadTranscriptView(e.target.value);
  });

  // Clear Chat
  document.getElementById("clearChatBtn").addEventListener("click", () => {
    document.getElementById("chatConversation").innerHTML = `
      <div class="chat-welcome-card">
        <div class="welcome-badge"><i class="fa-solid fa-graduation-cap"></i> YouTube NotebookLM Ready</div>
        <h2>Explore Your Video Knowledge Base</h2>
        <p>Ask in-depth questions across your ingested YouTube lectures, courses, and playlists.</p>
      </div>
    `;
  });

  // Open Keys Monitor
  document.getElementById("openKeyMonitorBtn").addEventListener("click", () => {
    switchStudioTab("keys");
  });

  // Rename Notebook Button
  document.getElementById("renameNotebookBtn").addEventListener("click", async () => {
    const titleEl = document.getElementById("currentNotebookTitle");
    const currentName = titleEl.textContent;
    const newName = prompt("Enter new Notebook Title:", currentName);
    if (newName && newName.trim() && newName.trim() !== currentName) {
      titleEl.textContent = newName.trim();
      // Update notebook title in DB if endpoint available or state
    }
  });
}
