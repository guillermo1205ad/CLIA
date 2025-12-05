// src/api.ts
const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8083";

export { BACKEND_URL };

export async function checkHealth() {
  const resp = await fetch(`${BACKEND_URL}/health`);
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}`);
  }
  return resp.json();
}

export interface AskResponse {
  answer: string;
  explanation?: any;
}

export async function askQuestion(question: string): Promise<AskResponse> {
  const form = new FormData();
  form.append("question", question);

  const resp = await fetch(`${BACKEND_URL}/query/text`, {
    method: "POST",
    body: form,
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }

  return resp.json();
}