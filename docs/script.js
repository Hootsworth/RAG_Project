const apiInput = document.getElementById("api-url");
const saveApi = document.getElementById("save-api");
const form = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const messages = document.getElementById("messages");
const evidenceList = document.getElementById("evidence-list");
const DEFAULT_API_URL = "https://rag-project-bufn.onrender.com";

apiInput.value = localStorage.getItem("rag_api_url") || DEFAULT_API_URL;

saveApi.addEventListener("click", () => {
  localStorage.setItem("rag_api_url", apiInput.value.trim().replace(/\/$/, ""));
  addBubble("BACKEND ENDPOINT SAVED. NEW QUESTIONS WILL USE THAT URL.", "bot");
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
    addBubble("ADD A BACKEND ENDPOINT FIRST. GITHUB PAGES RUNS THE INTERFACE. RENDER RUNS THE RAG API.", "bot");
    return;
  }
  if (!question) return;

  addBubble(question, "user");
  questionInput.value = "";
  const pending = addBubble("RETRIEVING TOPIC CHECKPOINTS AND MESSAGE CHUNKS. IF RENDER IS ASLEEP, THIS CAN TAKE A MOMENT.", "bot");

  try {
    const response = await fetch(`${apiUrl}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    pending.remove();
    addBubble(data.answer || data.error || "NO ANSWER RETURNED.", "bot");
    if (data.retrieval) renderEvidence(data.retrieval);
  } catch (error) {
    pending.remove();
    addBubble(`BACKEND UNREACHABLE: ${error.message}`, "bot");
  }
});
