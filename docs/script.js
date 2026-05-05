const apiInput = document.getElementById("api-url");
const saveApi = document.getElementById("save-api");
const form = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const messages = document.getElementById("messages");
const evidenceList = document.getElementById("evidence-list");

apiInput.value = localStorage.getItem("rag_api_url") || "";

saveApi.addEventListener("click", () => {
  localStorage.setItem("rag_api_url", apiInput.value.trim().replace(/\/$/, ""));
  addBubble("Backend URL saved.", "bot");
});

function addBubble(text, kind) {
  const bubble = document.createElement("article");
  bubble.className = `bubble ${kind}`;
  bubble.textContent = text;
  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
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

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const apiUrl = (localStorage.getItem("rag_api_url") || apiInput.value).trim().replace(/\/$/, "");
  const question = questionInput.value.trim();

  if (!apiUrl) {
    addBubble("Add your deployed Flask backend URL first. GitHub Pages can host this frontend, but it cannot run the Python RAG backend.", "bot");
    return;
  }
  if (!question) return;

  addBubble(question, "user");
  questionInput.value = "";
  const pending = addBubble("Retrieving checkpoints and message chunks...", "bot");

  try {
    const response = await fetch(`${apiUrl}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    pending.remove();
    addBubble(data.answer || data.error || "No answer returned.", "bot");
    if (data.retrieval) renderEvidence(data.retrieval);
  } catch (error) {
    pending.remove();
    addBubble(`Could not reach the backend: ${error.message}`, "bot");
  }
});
