const apiInput = document.getElementById("api-url");
const saveApi = document.getElementById("save-api");
const form = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const messages = document.getElementById("messages");
const evidenceList = document.getElementById("evidence-list");
const backendStatus = document.getElementById("backend-status");
const suggestionButtons = document.querySelectorAll("[data-question]");
const DEFAULT_API_URL = "https://rag-project-bufn.onrender.com";

apiInput.value = localStorage.getItem("rag_api_url") || DEFAULT_API_URL;

saveApi.addEventListener("click", () => {
  localStorage.setItem("rag_api_url", apiInput.value.trim().replace(/\/$/, ""));
  addBubble("Backend endpoint saved. New questions will use that URL.", "bot", { copyable: true });
  checkBackend();
});

suggestionButtons.forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    questionInput.focus();
  });
});

function addBubble(text, kind, options = {}) {
  const bubble = document.createElement("article");
  bubble.className = `bubble ${kind}`;
  const content = document.createElement("div");
  content.className = "bubble-content";
  content.textContent = text;
  bubble.appendChild(content);

  if (options.meta) {
    const meta = document.createElement("div");
    meta.className = "answer-meta";
    meta.textContent = options.meta;
    bubble.appendChild(meta);
  }

  if (options.copyable) {
    const copy = document.createElement("button");
    copy.className = "copy-answer";
    copy.type = "button";
    copy.textContent = "Copy";
    copy.addEventListener("click", async () => {
      await navigator.clipboard.writeText(text);
      copy.textContent = "Copied";
      setTimeout(() => {
        copy.textContent = "Copy";
      }, 1200);
    });
    bubble.appendChild(copy);
  }

  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

function setBubbleText(bubble, text) {
  const content = bubble.querySelector(".bubble-content");
  if (content) content.textContent = text;
}

function sourceMeta(retrieval) {
  const topics = retrieval?.topics?.length || 0;
  const chunks = retrieval?.chunks?.length || 0;
  const hundreds = retrieval?.hundred_checkpoints?.length || 0;
  return `Used ${topics} topic checkpoints, ${chunks} message chunks, ${hundreds} 100-message summaries`;
}

function renderEvidence(retrieval) {
  evidenceList.innerHTML = "";
  const items = [
    ...(retrieval.topics || []).slice(0, 3).map((item) => ({
      title: `Topic ${item.id} · messages ${item.start_message_id}-${item.end_message_id} · score ${item.score}`,
      body: item.summary,
    })),
    ...(retrieval.chunks || []).slice(0, 3).map((item) => ({
      title: `Chunk ${item.id} · messages ${item.start_message_id}-${item.end_message_id} · score ${item.score}`,
      body: item.text,
    })),
  ];

  if (!items.length) {
    evidenceList.textContent = "No evidence returned.";
    return;
  }

  items.forEach((item) => {
    const block = document.createElement("div");
    block.className = "evidence-item";
    const title = document.createElement("strong");
    const body = document.createElement("div");
    title.textContent = item.title;
    body.textContent = item.body;
    block.append(title, body);
    evidenceList.appendChild(block);
  });
}

async function checkBackend() {
  const apiUrl = (localStorage.getItem("rag_api_url") || apiInput.value || DEFAULT_API_URL).trim().replace(/\/$/, "");
  backendStatus.className = "backend-status checking";
  backendStatus.textContent = "Checking backend...";
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 12000);
    const response = await fetch(`${apiUrl}/health`, { signal: controller.signal });
    clearTimeout(timeout);
    if (!response.ok) throw new Error("Health check failed");
    backendStatus.className = "backend-status ready";
    backendStatus.textContent = "Backend ready";
  } catch (error) {
    backendStatus.className = "backend-status waking";
    backendStatus.textContent = "Backend may be waking";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const apiUrl = (localStorage.getItem("rag_api_url") || apiInput.value).trim().replace(/\/$/, "");
  const question = questionInput.value.trim();

  if (!apiUrl) {
    addBubble("Add a backend endpoint first. GitHub Pages runs the interface. Render runs the RAG API.", "bot", { copyable: true });
    return;
  }
  if (!question) return;

  addBubble(question, "user");
  questionInput.value = "";
  const pending = addBubble("Waking Render backend...", "bot");
  const loadingTimer = setTimeout(() => {
    setBubbleText(pending, "Searching conversation index...");
  }, 5000);

  try {
    const response = await fetch(`${apiUrl}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    clearTimeout(loadingTimer);
    pending.remove();
    addBubble(data.answer || data.error || "No answer returned.", "bot", {
      copyable: true,
      meta: data.retrieval ? sourceMeta(data.retrieval) : "",
    });
    if (data.retrieval) renderEvidence(data.retrieval);
  } catch (error) {
    clearTimeout(loadingTimer);
    pending.remove();
    addBubble(`Backend unreachable: ${error.message}`, "bot", { copyable: true });
  }
});

checkBackend();
