# CLIA – Conversa con tu grafo de documentos

CLIA es una interfaz tipo ChatGPT para **conversar con un grafo de documentos** construido con RAG‑Anything + LightRAG y un modelo de lenguaje local (`gpt-oss-20b` vía vLLM).

La versión **2.0** reemplaza el frontend en Streamlit por un frontend en **React + Vite**, manteniendo el backend FastAPI y el script de arranque con `tmux`.

---

## ✨ Características principales

- Chat estilo ChatGPT (burbuja usuario / asistente).
- **Soporte multimodal**: puedes adjuntar una imagen con un botón «+» y enviarla junto a tu pregunta.
- Panel lateral de **fuentes** con los documentos utilizados en la última respuesta.
- Indicador visual de **estado del backend** (punto verde/rojo).
- Renderizado de respuestas en **Markdown** (títulos, listas, negritas, etc.).
- Integración con:
  - Backend FastAPI (`backend/main.py`).
  - RAG‑Anything + LightRAG (modo `mix`).
  - Modelo local `gpt-oss-20b` servido vía endpoint OpenAI‑compatible (vLLM).
  - Reranker opcional `mxbai-rerank-large-v2`.

---

## 🗂 Estructura del repositorio

```text
CLIA/
├── backend/
│   └── main.py              # Backend FastAPI (RAG, LLM, descargas, health, etc.)
├── frontend-react/          # Frontend en React + Vite
│   ├── src/
│   │   ├── App.tsx          # Componente principal del chat
│   │   └── main.tsx         # Punto de entrada de React
│   ├── public/
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
├── frontend-streamlit/      # Versión antigua del frontend (Streamlit) – opcional
├── start_app.sh             # Script que levanta LLM + backend + frontend (tmux)
└── README.md
```

---

## 🔧 Requisitos

### Backend

- Python 3.10+ (probado en entorno tipo `multimodal_graph_env`).
- Dependencias típicas:
  - `fastapi`, `uvicorn`
  - `torch`, `sentence-transformers`, `transformers`
  - `raganything`, `lightrag`
- vLLM instalado para servir `gpt-oss-20b`.

> En tu entorno real (por ejemplo `~/multimodal_graph_env`) ya deberías tener instalados estos paquetes según tu proyecto de RAG.

### Frontend

- Node.js 18+ (recomendado LTS).
- `npm` o `pnpm`.

---

## ⚙️ Configuración del backend

El backend se encuentra en `backend/main.py`.  
Variables importantes:

- `TEXT_MODEL` – nombre lógico del modelo de lenguaje (`"gpt-oss-20b"`).
- `BASE_URL_TEXT` – URL del servidor OpenAI‑compatible de vLLM, por ejemplo:

  ```python
  BASE_URL_TEXT = "http://localhost:8010/v1"
  ```

- `STORAGE_PATH` – carpeta donde RAG‑Anything guarda el índice / grafo:

  ```bash
  export STORAGE_PATH=/home/gperalta/datos_grafo/rag_storage
  ```

- `RERANK_MODEL_PATH` – ruta al modelo de reranqueo (opcional):

  ```bash
  export RERANK_MODEL_PATH=/home/gperalta/models/mxbai-rerank-large-v2
  export ENABLE_RERANK=1   # o 0 para desactivar
  ```

El backend expone, entre otros:

- `GET /health` – estado del modelo y paths.
- `POST /query/text` – pregunta solo con texto.
- `POST /query/multimodal` – pregunta con texto + imagen.
- `GET /graph/data` – exporta el grafo de LightRAG.
- Endpoints de descarga: `/download/...`, `/download_path`.

Para ejecutarlo directamente (sin script):

```bash
cd ~/CLIA/backend
uvicorn main:app --host 0.0.0.0 --port 8083
```

---

## ⚙️ Configuración del frontend React

### 1. Variables de entorno (Vite)

El frontend lee la URL del backend desde `VITE_BACKEND_INTERNAL_URL`.

Crea un archivo `.env` dentro de `frontend-react/` (o usa tus propias variables):

```bash
cd ~/CLIA/frontend-react
cat << 'EOF' > .env
VITE_BACKEND_INTERNAL_URL=http://localhost:8083
EOF
```

> Si sirves el backend detrás de un reverse proxy / dominio público, apunta esta URL a ese endpoint.

### 2. Instalar dependencias

```bash
cd ~/CLIA/frontend-react
npm install
```

(Ya incluye `react-markdown` y `remark-gfm` para mostrar bien el Markdown de las respuestas.)

### 3. Ejecutar en modo desarrollo

```bash
npm run dev -- --host 0.0.0.0 --port 8502
```

Luego entra en el navegador a:

- `http://localhost:8502` (o la URL/puerto que corresponda).

---

## 🚀 Arranque integrado con `start_app.sh`

El script `start_app.sh` lanza:

1. vLLM con `gpt-oss-20b` en un puerto (p.ej. 8010).
2. El backend FastAPI en el puerto `8083`.
3. El frontend (Streamlit o React, según la versión del script) en el puerto `8502`.

Uso típico:

```bash
cd ~/CLIA
bash start_app.sh 2   # 2 = índice de la GPU física
```

El script:

- Configura variables como `BACKEND_INTERNAL_URL` y `BACKEND_BROWSER_URL`.
- Crea sesiones `tmux` llamadas, por ejemplo, `gptoss20b`, `clia_backend`, `clia_frontend`.
- Escribe logs en `~/logs_memoria/`.

Para detener los servicios:

```bash
tmux kill-session -t gptoss20b
tmux kill-session -t clia_backend
tmux kill-session -t clia_frontend
```

---

## 🧠 Flujo de uso (frontend React)

1. El usuario escribe una pregunta en el campo de texto.
2. Opcionalmente, adjunta una imagen con el botón «+» (imagen se muestra abajo con opción de quitarla).
3. Al enviar:
   - Se deshabilita el input y aparece un **spinner** “Pensando…”.
   - El frontend hace:
     - `POST /query/multimodal` si hay imagen.
     - `POST /query/text` si solo hay texto.
4. El backend genera la respuesta y devuelve:
   - `answer` (texto en Markdown).
   - `explanation.documents` (lista de PDFs asociados a la respuesta).
5. El panel izquierdo muestra los documentos usados como **fuentes**.

---

## 🧪 Comprobaciones rápidas

- **Backend**:

  ```bash
  curl -s http://localhost:8083/health | jq
  ```

- **LLM vLLM**:

  ```bash
  curl -s http://localhost:8010/v1/models | jq
  ```

- **Frontend**: abre en el navegador la URL donde corre Vite (`http://localhost:8502`).

---

## 📌 Notas

- Este repositorio asume que el índice RAG y los documentos ya existen en rutas como:

  - `/home/gperalta/datos_grafo/rag_storage`
  - `/home/gperalta/datos_grafo/docs`
  - `/home/gperalta/datos_grafo/output_rag`

- Para entornos nuevos, necesitarás:
  - Construir el índice con RAG‑Anything / LightRAG.
  - Ajustar las rutas (`STORAGE_PATH`, `DOCS_PATHS`, etc.) en `backend/main.py` y/o variables de entorno.

---

## 📝 Licencia

Puedes adaptar esta sección a la licencia que quieras usar (por ejemplo, MIT, Apache‑2.0 o una licencia académica específica).

