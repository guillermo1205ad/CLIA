import React, { useEffect, useState } from "react";

/** Tipos básicos */

type Role = "user" | "assistant";

interface Message {
  id: number;
  role: Role;
  content: string;
  image?: string | null; // data URL para preview en el chat
}

/** URLs de backend desde variables Vite (o fallback local) */

const BACKEND_INTERNAL_URL: string =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ((import.meta as any).env?.VITE_BACKEND_INTERNAL_URL as string) ||
  "http://localhost:8083";

/* ------------------------------------------------------------------
   Renderizador MUY sencillo de Markdown (títulos, listas, **negrita**)
   ------------------------------------------------------------------ */

/** Procesa negritas **texto** dentro de una línea */
function renderInline(text: string): (string | JSX.Element)[] {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, idx) => {
    const m = part.match(/^\*\*([^*]+)\*\*$/);
    if (m) {
      return (
        <strong key={idx} style={{ fontWeight: 600 }}>
          {m[1]}
        </strong>
      );
    }
    return part;
  });
}

/** Convierte un bloque de texto markdown a JSX */
const MarkdownBlock: React.FC<{ text: string }> = ({ text }) => {
  const lines = text.split(/\r?\n/);
  const elements: JSX.Element[] = [];
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    let line = lines[i];

    // Saltar líneas vacías
    if (!line.trim()) {
      i++;
      continue;
    }

    // Regla horizontal "---"
    if (/^---\s*$/.test(line.trim())) {
      elements.push(
        <hr
          key={key++}
          style={{
            border: "none",
            borderTop: "1px solid rgba(148,163,184,0.25)",
            margin: "0.6rem 0",
          }}
        />
      );
      i++;
      continue;
    }

    // #### H4 (subtítulo)
    if (/^####\s+/.test(line)) {
      const content = line.replace(/^####\s+/, "").trim();
      elements.push(
        <h4
          key={key++}
          style={{
            fontSize: 14,
            fontWeight: 600,
            margin: "0.45rem 0 0.25rem",
            color: "#9ca3af",
          }}
        >
          {renderInline(content)}
        </h4>
      );
      i++;
      continue;
    }

    // ### H3
    if (/^###\s+/.test(line)) {
      const content = line.replace(/^###\s+/, "").trim();
      elements.push(
        <h3
          key={key++}
          style={{
            fontSize: 16,
            fontWeight: 600,
            margin: "0.5rem 0 0.3rem",
          }}
        >
          {renderInline(content)}
        </h3>
      );
      i++;
      continue;
    }

    // ## H2
    if (/^##\s+/.test(line)) {
      const content = line.replace(/^##\s+/, "").trim();
      elements.push(
        <h2
          key={key++}
          style={{
            fontSize: 18,
            fontWeight: 600,
            margin: "0.55rem 0 0.35rem",
          }}
        >
          {renderInline(content)}
        </h2>
      );
      i++;
      continue;
    }

    // # H1
    if (/^#\s+/.test(line)) {
      const content = line.replace(/^#\s+/, "").trim();
      elements.push(
        <h1
          key={key++}
          style={{
            fontSize: 20,
            fontWeight: 700,
            margin: "0 0 0.45rem",
          }}
        >
          {renderInline(content)}
        </h1>
      );
      i++;
      continue;
    }

    // Listas con - o *
    if (/^[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^[-*]\s+/, "").trim());
        i++;
      }
      elements.push(
        <ul
          key={key++}
          style={{
            margin: "0.1rem 0 0.45rem 1.2rem",
            padding: 0,
          }}
        >
          {items.map((it, idx) => (
            <li key={idx} style={{ marginBottom: 2 }}>
              {renderInline(it)}
            </li>
          ))}
        </ul>
      );
      continue;
    }

    // Párrafo (acumulamos hasta una línea vacía)
    const paraLines: string[] = [];
    while (i < lines.length && lines[i].trim() !== "") {
      paraLines.push(lines[i]);
      i++;
    }
    const paraText = paraLines.join(" ");
    elements.push(
      <p
        key={key++}
        style={{
          margin: "0 0 0.45rem",
        }}
      >
        {renderInline(paraText)}
      </p>
    );
  }

  return <>{elements}</>;
};

function App() {
  const [backendOk, setBackendOk] = useState<boolean>(false);
  const [backendStatusText, setBackendStatusText] = useState<string>(
    "Comprobando backend…"
  );

  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      role: "assistant",
      content:
        "¿Qué vamos a explorar hoy? Pregúntame algo sobre tus documentos.",
    },
  ]);

  const [input, setInput] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const [refs, setRefs] = useState<string[]>([]);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  /** ---------- CSS global para el spinner ---------- */
  useEffect(() => {
    const style = document.createElement("style");
    style.innerHTML = `
      @keyframes clia-spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }
    `;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  /** ---------- HEALTH DEL BACKEND ---------- */

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const resp = await fetch(`${BACKEND_INTERNAL_URL}/health`, {
          method: "GET",
        });
        if (!resp.ok) {
          setBackendOk(false);
          setBackendStatusText(`Error HTTP ${resp.status}`);
          return;
        }
        const data = await resp.json();
        if (data?.status === "ok") {
          setBackendOk(true);
          setBackendStatusText("Backend conectado");
        } else {
          setBackendOk(false);
          setBackendStatusText("Respuesta inesperada del backend");
        }
      } catch (err: any) {
        setBackendOk(false);
        setBackendStatusText(`No se pudo conectar al backend: ${err?.message}`);
      }
    };

    checkHealth();
  }, []);

  /** ---------- MANEJO DE IMAGEN ---------- */

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      setImageFile(null);
      setImagePreview(null);
      return;
    }
    setImageFile(file);

    const reader = new FileReader();
    reader.onload = () => {
      setImagePreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const clearImage = () => {
    setImageFile(null);
    setImagePreview(null);
  };

  /** ---------- ENVÍO DE MENSAJE ---------- */

  const handleSend = async () => {
    const question = input.trim();

    if (!question && !imageFile) return;
    if (!backendOk) return;

    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        role: "user",
        content: question || "(imagen sin texto)",
        image: imagePreview,
      },
    ]);

    setInput("");
    setImagePreview(null);

    setIsLoading(true);
    try {
      let response: Response;

      if (imageFile) {
        const form = new FormData();
        form.append("question", question || "(sin texto)");
        form.append("image", imageFile);

        response = await fetch(`${BACKEND_INTERNAL_URL}/query/multimodal`, {
          method: "POST",
          body: form,
        });

        setImageFile(null);
      } else {
        const body = new URLSearchParams({ question });
        response = await fetch(`${BACKEND_INTERNAL_URL}/query/text`, {
          method: "POST",
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
          },
          body,
        });
      }

      if (!response.ok) {
        const txt = await response.text();
        throw new Error(`HTTP ${response.status}: ${txt}`);
      }

      const data = await response.json();
      const rawAnswer = (data?.answer || "").trim();
      const explanation = data?.explanation || {};
      const docs: string[] = explanation?.documents || [];

      setRefs(Array.isArray(docs) ? docs : []);

      const answer =
        rawAnswer ||
        "⚠️ No se encontró información relacionada en el grafo.";

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "assistant",
          content: answer,
        },
      ]);
    } catch (err: any) {
      const msg = `❌ Error al conectar con el backend: ${
        err?.message || String(err)
      }`;
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "assistant",
          content: msg,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  /** ---------- ESC para limpiar imagen ---------- */

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        clearImage();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  /** ---------- RENDER ---------- */

  const statusColor = backendOk ? "#22c55e" : "#f97373";

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "radial-gradient(circle at top, #111827, #020617)",
        color: "#e5e7eb",
        fontFamily:
          "-apple-system,BlinkMacSystemFont,system-ui,Segoe UI,Helvetica,Arial,sans-serif",
        padding: "0.75rem 1.25rem 1rem",
        boxSizing: "border-box",
      }}
    >
      {/* HEADER SUPERIOR */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "0.75rem",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div
            style={{
              width: 34,
              height: 34,
              borderRadius: 12,
              background:
                "linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontWeight: 700,
              fontSize: 18,
            }}
          >
            C
          </div>
          <div>
            <div
              style={{
                fontSize: 18,
                fontWeight: 600,
                letterSpacing: 0.4,
              }}
            >
              CLIA
            </div>
            <div
              style={{
                fontSize: 11,
                color: "#9ca3af",
                marginTop: -2,
              }}
            >
              Conversa con tu grafo de documentos
            </div>
          </div>
        </div>

        <div
          style={{
            fontSize: 12,
            color: "#9ca3af",
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <span
            style={{
              display: "inline-block",
              width: 10,
              height: 10,
              borderRadius: "999px",
              backgroundColor: statusColor,
              boxShadow: backendOk
                ? "0 0 8px rgba(34,197,94,0.6)"
                : "0 0 8px rgba(248,113,113,0.6)",
            }}
          />
          {backendStatusText}
        </div>
      </div>

      {/* GRID PRINCIPAL */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(260px, 320px) minmax(0, 1fr)",
          gap: "1.5rem",
          height: "calc(100vh - 70px)",
        }}
      >
        {/* --------- FUENTES --------- */}
        <div
          style={{
            borderRadius: 16,
            background: "rgba(15,23,42,0.98)",
            border: "1px solid rgba(148,163,184,0.18)",
            padding: "1rem 1.1rem",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
            FUENTES
          </div>
          <div
            style={{
              fontSize: 11,
              color: "#9ca3af",
              marginBottom: 10,
            }}
          >
            (En esta versión mínima aún no mostramos referencias directas.)
          </div>

          <div
            style={{
              flex: 1,
              overflowY: "auto",
              fontSize: 13,
              paddingRight: 4,
            }}
          >
            {refs.length === 0 ? (
              <div
                style={{
                  fontSize: 13,
                  color: "#6b7280",
                  marginTop: 6,
                }}
              >
                No hay fuentes asociadas a la última respuesta.
              </div>
            ) : (
              <ul
                style={{
                  listStyle: "none",
                  padding: 0,
                  margin: 0,
                  display: "flex",
                  flexDirection: "column",
                  gap: 4,
                }}
              >
                {refs.slice(0, 30).map((name, idx) => (
                  <li
                    key={`${idx}-${name}`}
                    style={{
                      padding: "6px 8px",
                      borderRadius: 8,
                      backgroundColor: "rgba(15,23,42,0.8)",
                      border: "1px solid rgba(51,65,85,0.8)",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                    title={name}
                  >
                    <span
                      style={{
                        color: "#9ca3af",
                        marginRight: 6,
                        fontSize: 11,
                      }}
                    >
                      [{idx + 1}]
                    </span>
                    {name}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* --------- CHAT --------- */}
        <div
          style={{
            borderRadius: 16,
            background: "rgba(15,23,42,0.98)",
            border: "1px solid rgba(148,163,184,0.18)",
            display: "flex",
            flexDirection: "column",
          }}
        >
          {/* Área de mensajes */}
          <div
            style={{
              flex: 1,
              overflowY: "auto",
              padding: "1.1rem 1.3rem 0.6rem",
              display: "flex",
              flexDirection: "column",
              gap: 10,
            }}
          >
            {messages.map((m) => {
              const isUser = m.role === "user";
              return (
                <div
                  key={m.id}
                  style={{
                    display: "flex",
                    justifyContent: isUser ? "flex-end" : "flex-start",
                  }}
                >
                  <div
                    style={{
                      maxWidth: "80%",
                      borderRadius: 18,
                      padding: "0.7rem 0.9rem",
                      backgroundColor: isUser
                        ? "rgba(37,99,235,0.9)"
                        : "rgba(15,23,42,0.95)",
                      color: isUser ? "#e5e7eb" : "#e5e7eb",
                      fontSize: 14,
                      lineHeight: 1.5,
                      boxShadow: isUser
                        ? "0 4px 16px rgba(37,99,235,0.35)"
                        : "0 4px 16px rgba(15,23,42,0.8)",
                    }}
                  >
                    {m.image && (
                      <img
                        src={m.image}
                        alt="adjunta"
                        style={{
                          maxWidth: "100%",
                          borderRadius: 12,
                          marginBottom: 6,
                        }}
                      />
                    )}

                    {/* Markdown simple */}
                    <MarkdownBlock text={m.content} />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Indicador de carga */}
          {isLoading && (
            <div
              style={{
                padding: "0 1.3rem 0.4rem",
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <div
                style={{
                  width: 16,
                  height: 16,
                  borderRadius: "999px",
                  border: "2px solid rgba(148,163,184,0.4)",
                  borderTopColor: "#60a5fa",
                  animation: "clia-spin 0.7s linear infinite",
                }}
              />
              <span style={{ fontSize: 12, color: "#9ca3af" }}>Pensando…</span>
            </div>
          )}

          {/* Preview de imagen preparada */}
          {imagePreview && (
            <div
              style={{
                padding: "0 1.3rem 0.3rem",
                display: "flex",
                alignItems: "center",
                gap: 10,
              }}
            >
              <div
                style={{
                  position: "relative",
                  width: 100,
                  height: 70,
                  borderRadius: 12,
                  overflow: "hidden",
                  border: "1px solid rgba(55,65,81,0.85)",
                  flexShrink: 0,
                }}
              >
                <img
                  src={imagePreview}
                  alt="preview"
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                  }}
                />
                <button
                  onClick={clearImage}
                  title="Quitar imagen"
                  style={{
                    position: "absolute",
                    top: 4,
                    right: 4,
                    width: 20,
                    height: 20,
                    borderRadius: "999px",
                    border: "none",
                    backgroundColor: "rgba(15,23,42,0.9)",
                    color: "#e5e7eb",
                    fontSize: 12,
                    lineHeight: 1,
                    cursor: "pointer",
                  }}
                >
                  ✕
                </button>
              </div>
              <div style={{ fontSize: 12, color: "#9ca3af" }}>
                Imagen preparada. Se enviará junto con tu próxima pregunta.
                <br />
                <span style={{ opacity: 0.8 }}>
                  (Haz clic en ✕ o presiona Esc para quitarla)
                </span>
              </div>
            </div>
          )}

          {/* Barra inferior: +  / input / botón enviar */}
          <div
            style={{
              padding: "0.5rem 1.1rem 0.9rem",
              display: "flex",
              alignItems: "center",
              gap: "0.8rem",
            }}
          >
            {/* Botón + redondo */}
            <label
              style={{
                position: "relative",
                width: 40,
                height: 40,
                borderRadius: "999px",
                border: "1px solid #4b5563",
                backgroundColor: "#020617",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                flexShrink: 0,
              }}
            >
              <span
                style={{
                  color: "#e5e7eb",
                  fontSize: 26,
                  marginTop: -2,
                  pointerEvents: "none",
                }}
              >
                +
              </span>
              <input
                type="file"
                accept="image/*"
                style={{
                  position: "absolute",
                  inset: 0,
                  opacity: 0,
                  cursor: "pointer",
                }}
                onChange={handleImageChange}
              />
            </label>

            {/* Input de texto */}
            <input
              type="text"
              placeholder={
                backendOk
                  ? "Escribe tu pregunta…"
                  : "Backend no disponible en este momento"
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              disabled={!backendOk || isLoading}
              style={{
                flex: 1,
                borderRadius: 999,
                border: "1px solid #4b5563",
                backgroundColor: "#020617",
                color: "#e5e7eb",
                padding: "0.65rem 1rem",
                outline: "none",
                fontSize: 14,
              }}
            />

            {/* Botón enviar */}
            <button
              onClick={handleSend}
              disabled={
                !backendOk || isLoading || (!input.trim() && !imageFile)
              }
              style={{
                width: 44,
                height: 44,
                borderRadius: "999px",
                border: "none",
                backgroundColor:
                  !backendOk || (!input.trim() && !imageFile)
                    ? "#374151"
                    : "#2563eb",
                color: "#e5e7eb",
                cursor:
                  !backendOk || (!input.trim() && !imageFile)
                    ? "not-allowed"
                    : "pointer",
                fontSize: 18,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
                boxShadow:
                  !backendOk || (!input.trim() && !imageFile)
                    ? "none"
                    : "0 4px 14px rgba(37,99,235,0.55)",
              }}
            >
              {isLoading ? "…" : "➤"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;