#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend de CLIA 🇨🇱

- RAGAnything + LightRAG con el mismo storage (mode="mix")
- Reranker local: mixedbread-ai/mxbai-rerank-large-v2
- Explicabilidad con enlaces PDF.js y control de archivos privados
- Endpoint robusto /download con búsqueda flexible
"""

import os
import asyncio
import unicodedata
import re
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote

import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache


# ─────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TEXT_MODEL = "gpt-oss-20b"
BASE_URL_TEXT = "http://localhost:8010/v1"
API_KEY_TEXT = "none"

STORAGE_PATH = os.environ.get("STORAGE_PATH", "path")
RERANK_MODEL_PATH = os.environ.get("RERANK_MODEL_PATH", "path")

ENABLE_RERANK = os.environ.get("ENABLE_RERANK", "0").lower() in ("1", "true", "yes")

DOCS_PATHS = [
    "path",
]
DOCS_MOUNTS = [f"/files/{i}" for i in range(len(DOCS_PATHS))]
PDFJS_VIEWER = "https://mozilla.github.io/pdf.js/web/viewer.html"

rag_instance = None
reranker_tokenizer = None
reranker_model = None

app = FastAPI(title="CLIA Backend", version="3.2")

# Monta documentos estáticos
for base, mount in zip(DOCS_PATHS, DOCS_MOUNTS):
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    app.mount(mount, StaticFiles(directory=str(p), html=False), name=f"docs_{p.name}")

# CORS global
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# LLM + EMBEDDINGS
# ─────────────────────────────────────────────
def build_llm_func():
    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return openai_complete_if_cache(
            TEXT_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=API_KEY_TEXT,
            base_url=BASE_URL_TEXT,
            **kwargs,
        )
    return llm_model_func


def build_embedding_func():
    model_id = "intfloat/multilingual-e5-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Embeddings] Cargando {model_id} en {device.upper()}")
    embedder = SentenceTransformer(model_id, device=device)

    async def _embed_async(texts):
        def _encode():
            vecs = embedder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return vecs.tolist()
        return await asyncio.to_thread(_encode)

    return EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=_embed_async)

# ─────────────────────────────────────────────
# RERANKER
# ─────────────────────────────────────────────
def ensure_reranker_loaded():
    global reranker_tokenizer, reranker_model
    if not ENABLE_RERANK:
        return
    if reranker_model is not None:
        return
    print(f"[Rerank] Cargando modelo: {RERANK_MODEL_PATH}")
    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_PATH)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        RERANK_MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).eval()
    if torch.cuda.is_available():
        reranker_model.cuda()
    print("✅ Reranker cargado (mxbai-rerank-large-v2)")


def rerank_chunks(question: str, chunks: list[dict], top_k: int = 100) -> list[dict]:
    if not chunks:
        return []
    if not ENABLE_RERANK:
        return chunks[:top_k]
    ensure_reranker_loaded()
    texts = [c.get("text", "") for c in chunks]
    q_list = [question] * len(texts)
    with torch.no_grad():
        enc = reranker_tokenizer(q_list, texts, padding=True, truncation=True, return_tensors="pt")
        if torch.cuda.is_available():
            enc = {k: v.cuda() for k, v in enc.items()}
        logits = reranker_model(**enc).logits
        probs = F.softmax(logits, dim=-1)
        scores = probs[:, 1].detach().cpu().tolist()
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]

# ─────────────────────────────────────────────
# RAGAnything INIT
# ─────────────────────────────────────────────
async def get_rag():
    global rag_instance
    if rag_instance is None:
        print("⚙️ Inicializando RAG-Anything…")
        cfg = RAGAnythingConfig(
            working_dir=STORAGE_PATH,
            context_window=100,
            context_mode="page",
            max_context_tokens=32000,
            parse_method="auto",
        )
        rag_instance = RAGAnything(
            config=cfg,
            llm_model_func=build_llm_func(),
            embedding_func=build_embedding_func(),
        )
        await rag_instance._ensure_lightrag_initialized()
        if getattr(rag_instance, "lightrag", None):
            rag_instance.lightrag.storage_root = STORAGE_PATH
        print(f"✅ RAG listo en {STORAGE_PATH}")
    return rag_instance

# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────
def is_private(path_str: str) -> bool:
    try:
        return Path(path_str).name.startswith("priv.")
    except Exception:
        return True

def public_file_url(abs_path: str) -> str | None:
    p = Path(abs_path)
    for base, mount in zip(DOCS_PATHS, DOCS_MOUNTS):
        rb = Path(base).resolve()
        try:
            rp = p.resolve()
            if str(rp).startswith(str(rb)):
                rel = rp.relative_to(rb)
                return f"{mount}/{quote(str(rel))}"
        except Exception:
            continue
    return None

def search_terms_from_text(text: str, max_terms: int = 6) -> str:
    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{4,}", text)[:max_terms]
    return " ".join(words)

def try_extract_context(rag) -> dict:
    lr = getattr(rag, "lightrag", None)
    if lr is None:
        return {}
    for attr in ("last_context", "context", "latest_context"):
        ctx = getattr(lr, attr, None)
        if ctx:
            return ctx
    return {}

def enrich_chunks_for_ui(question: str, chunks: list[dict]) -> list[dict]:
    out = []
    for i, ch in enumerate(chunks, 1):
        text = ch.get("text", "") or ch.get("content", "")
        src = ch.get("source") or ch.get("file_path") or ch.get("path") or ""
        source_url, viewer_url = None, None
        if src and not is_private(src):
            murl = public_file_url(src)
            if murl:
                source_url = f"http://localhost:8083{murl}"
                if str(src).lower().endswith(".pdf"):
                    terms = search_terms_from_text(text)
                    viewer_url = f"{PDFJS_VIEWER}?file={quote(source_url)}#search={quote(terms)}"
        out.append({
            "id": f"C{i}",
            "text": text,
            "source_name": Path(src).name if src else "",
            "source_url": source_url,
            "viewer_url": viewer_url,
        })
    return out

# ─────────────────────────────────────────────
# ENDPOINTS PRINCIPALES
# ─────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": TEXT_MODEL,
        "storage": STORAGE_PATH,
        "reranker_loaded": bool(reranker_model),
        "rerank_enabled": ENABLE_RERANK,
    }

@app.post("/query/text")
async def query_text(question: str = Form(...)):
    try:
        rag = await get_rag()
        print(f"🔎 Q: {question}")
        answer = await rag.aquery(question, mode="mix")

        ctx = try_extract_context(rag)
        raw_chunks = ctx.get("chunks") or ctx.get("naive_context") or []
        relations = ctx.get("relations") or []

        visible = [c for c in raw_chunks if not is_private(c.get("source", ""))]
        top_chunks = rerank_chunks(question, visible, 200)
        enriched = enrich_chunks_for_ui(question, top_chunks)
        documents = sorted({Path(e["source_name"]).name for e in enriched if e["source_name"]})

        explanation = {"chunks": enriched, "relations": relations, "documents": documents}
        return JSONResponse({"answer": answer, "explanation": explanation})
    except Exception as e:
        print(f"❌ /query/text: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/query/multimodal")
async def query_multimodal(question: str = Form(...), image: UploadFile | None = None):
    return await query_text(question)

@app.get("/graph/data")
async def graph_data():
    try:
        rag = await get_rag()
        graph = await rag.lightrag.export_graph()
        return JSONResponse(graph)
    except Exception as e:
        print(f"⚠️ /graph/data: {e}")
        return JSONResponse({"nodes": [], "links": []})

@app.get("/doc/view/{mount_id}/{path:path}")
async def doc_view(mount_id: int, path: str, q: str = ""):
    try:
        if Path(path).name.startswith("priv."):
            return JSONResponse({"error": "Documento privado"}, status_code=403)
        file_abs = f"http://localhost:8083{DOCS_MOUNTS[mount_id]}/{quote(path)}"
        viewer = f"{PDFJS_VIEWER}?file={quote(file_abs)}"
        if q:
            viewer += f"#search={quote(q)}"
        return RedirectResponse(viewer)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ─────────────────────────────────────────────
# DESCARGA ROBUSTA
# ─────────────────────────────────────────────
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9_.]+", "_", s)
    return s

def _best_match_file(mount_id: int, filename: str) -> Path | None:
    base = Path(DOCS_PATHS[mount_id]).resolve()
    if not base.exists():
        return None
    target = _norm(filename)
    best, best_score = None, 0.0
    for p in base.rglob("*"):
        if not p.is_file() or p.name.startswith("priv."):
            continue
        score = SequenceMatcher(None, _norm(p.name), target).ratio()
        if score > best_score:
            best, best_score = p, score
        if best_score > 0.96:
            break
    return best if best_score > 0.6 else None

@app.get("/download/{mount_id}/{filename:path}")
async def download_file(mount_id: int, filename: str):
    try:
        if Path(filename).name.startswith("priv."):
            return JSONResponse({"error": "Documento privado"}, status_code=403)
        found = _best_match_file(mount_id, filename)
        if not found:
            return JSONResponse({"error": f"No encontrado: {filename}"}, status_code=404)
        mime = "application/pdf" if found.suffix.lower() == ".pdf" else "application/octet-stream"
        print(f"⬇️ Descarga: {found.name} (match ok)")
        return FileResponse(str(found), media_type=mime, filename=found.name)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("🚀 Backend CLIA activo en http://0.0.0.0:8083")
    if ENABLE_RERANK:
        try:
            ensure_reranker_loaded()
        except Exception as e:
            print(f"⚠️ Reranker no cargado: {e}")
    uvicorn.run(app, host="0.0.0.0", port=8083, reload=False, log_config=None)