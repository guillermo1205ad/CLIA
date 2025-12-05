#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frontend CLIA – Estilo ChatGPT con botón "+" para imagen y estado sólo con círculo.
"""

import streamlit as st
import requests
import urllib.parse
import os
import base64
import re

# ─────────────────────────────────────────────
# CONFIG BÁSICA
# ─────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="CLIA")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8083")
BACKEND_INTERNAL_URL = os.getenv("BACKEND_INTERNAL_URL", BACKEND_URL)

# ─────────────────────────────────────────────
# CSS GLOBAL (incluye estilo del "+" redondo)
# ─────────────────────────────────────────────

st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    .main .block-container {
        padding-top: 0.6rem;
        padding-bottom: 0.4rem;
    }

    [data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
    }

    /* --- Botón "+" redondo (file_uploader) --- */
    .plus-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .plus-wrapper div[data-testid="stFileUploader"] {
        margin: 0;
        padding: 0;
        width: 40px;
    }

    .plus-wrapper div[data-testid="stFileUploader"] > div {
        margin: 0;
        padding: 0;
    }

    .plus-wrapper section[data-testid="stFileUploadDropzone"] {
        position: relative;
        width: 40px;
        height: 40px;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 999px !important;
        border: 1px solid #555 !important;
        background-color: #333 !important;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
    }

    /* Ocultar textos internos feos */
    .plus-wrapper section[data-testid="stFileUploadDropzone"] * {
        color: transparent !important;
        font-size: 0 !important;
    }

    /* El input ocupa todo el círculo pero no se ve */
    .plus-wrapper section[data-testid="stFileUploadDropzone"] input[type="file"] {
        position: absolute;
        width: 100%;
        height: 100%;
        opacity: 0;
        cursor: pointer;
        top: 0;
        left: 0;
    }

    /* Símbolo "+" encima de todo */
    .plus-wrapper section[data-testid="stFileUploadDropzone"]::after {
        content: "+";
        position: absolute;
        color: #fff;
        font-size: 1.6rem;
        font-weight: 400;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -55%);
        pointer-events: none;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# AUXILIARES
# ─────────────────────────────────────────────

def extract_pdf_refs_from_answer(answer: str) -> list[str]:
    if not answer:
        return []
    pattern = r"\[\d+\]\s+(.+?\.pdf)"
    return re.findall(pattern, answer)


def strip_references_section(answer: str) -> str:
    if not answer:
        return answer
    lines = answer.splitlines()
    cut_idx = None
    for i, line in enumerate(lines):
        norm = re.sub(r'^[#\\-*\\s]+', '', line).strip().lower()
        if norm.startswith("references") or norm.startswith("referencias"):
            cut_idx = i
            break
    if cut_idx is None:
        return answer
    kept = lines[:cut_idx]
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def fetch_pdf_bytes(backend_internal_url: str, filename: str) -> bytes | None:
    encoded = urllib.parse.quote(filename)
    url = f"{backend_internal_url}/download/0/{encoded}"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        return None
    return None


# ─────────────────────────────────────────────
# ESTADO INICIAL
# ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "¿Qué vamos a explorar hoy? Pregúntame algo sobre tus documentos.",
            "image": None,
        }
    ]

if "last_refs" not in st.session_state:
    st.session_state.last_refs = []

if "staged_image" not in st.session_state:
    st.session_state.staged_image = None

# ─────────────────────────────────────────────
# HEALTH DEL BACKEND (solo círculo)
# ─────────────────────────────────────────────

backend_ok = False
try:
    r = requests.get(f"{BACKEND_INTERNAL_URL}/health", timeout=5)
    backend_ok = (r.status_code == 200)
except Exception:
    backend_ok = False

status_color = "#4caf50" if backend_ok else "#f44336"  # verde / rojo

# ─────────────────────────────────────────────
# CABECERA (círculo + LOGO + TÍTULO)
# ─────────────────────────────────────────────

ASSISTANT_AVATAR = "static/icono.png"
USER_AVATAR = "static/user.png"

icon_b64 = ""
try:
    with open(ASSISTANT_AVATAR, "rb") as f:
        icon_b64 = base64.b64encode(f.read()).decode("utf-8")
except Exception:
    pass

top_col1, top_col2, top_col3 = st.columns([0.3, 2, 0.3])
with top_col1:
    st.markdown(
        f"<div style='height:2rem;display:flex;align-items:center;'><span style='color:{status_color};font-size:1.4rem;'>●</span></div>",
        unsafe_allow_html=True,
    )

with top_col2:
    if icon_b64:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:1rem;justify-content:center;margin-bottom:0.4rem;">
                <img src="data:image/png;base64,{icon_b64}" width="50" style="border-radius:50%;" />
                <h1 style="margin:0;">CLIA</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.title("CLIA")

st.markdown("---")

# ─────────────────────────────────────────────
# LAYOUT: FUENTES (IZQ) + CHAT (DER)
# ─────────────────────────────────────────────

col_sources, col_chat = st.columns([1, 3], gap="large")

# ───────────── Columna izquierda: FUENTES ─────────────
with col_sources:
    st.markdown("### Fuentes")

    refs = st.session_state.get("last_refs", [])
    if not refs:
        st.info("No hay fuentes asociadas a la última respuesta.")
    else:
        MAX_REFS = 15
        to_show = refs[:MAX_REFS]
        hidden = max(0, len(refs) - len(to_show))

        for i, filename in enumerate(to_show, 1):
            pdf_bytes = fetch_pdf_bytes(BACKEND_INTERNAL_URL, filename)
            label = f"[{i}] {filename}"
            if pdf_bytes:
                st.download_button(
                    label=label,
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    key=f"src_{i}_{filename}",
                    use_container_width=True,
                )
            else:
                st.button(label, key=f"src_{i}_{filename}_err", use_container_width=True)

        if hidden > 0:
            st.caption(f"… y {hidden} fuentes adicionales no mostradas.")

# ───────────── Columna derecha: CHAT ─────────────
with col_chat:
    if not backend_ok:
        st.warning("⚠️ El backend no está disponible. No se puede iniciar el chat.")
    else:
        # 1) Historial de mensajes
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                avatar = ASSISTANT_AVATAR if msg["role"] == "assistant" else USER_AVATAR
                with st.chat_message(msg["role"], avatar=avatar):
                    if msg.get("image") is not None:
                        st.image(msg["image"], width=220)
                    st.markdown(msg["content"])

        # 2) Preview de imagen preparada
        if st.session_state.staged_image is not None:
            prev_c1, prev_c2 = st.columns([1, 3])
            with prev_c1:
                st.image(
                    st.session_state.staged_image,
                    caption="Imagen preparada",
                    width=120,
                )
            with prev_c2:
                if st.button("Quitar imagen", key="remove_image"):
                    st.session_state.staged_image = None
                    st.experimental_rerun()

        # 3) Fila inferior: "+" (uploader) + barra de chat
        bottom_col_plus, bottom_col_input = st.columns([0.1, 0.9])

        with bottom_col_plus:
            st.markdown("<div class='plus-wrapper'>", unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Adjuntar imagen",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
                key="image_uploader",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if uploaded is not None:
                st.session_state.staged_image = uploaded.getvalue()

        with bottom_col_input:
            user_input = st.chat_input("Escribe tu pregunta")

        # 4) Envío del mensaje
        if user_input:
            staged_image_bytes = st.session_state.staged_image
            st.session_state.staged_image = None  # se consume

            # mensaje usuario
            with st.chat_message("user", avatar=USER_AVATAR):
                if staged_image_bytes is not None:
                    st.image(staged_image_bytes, width=220)
                st.markdown(user_input)

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_input,
                    "image": staged_image_bytes,
                }
            )

            # respuesta asistente
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                with st.spinner("Pensando…"):
                    try:
                        if staged_image_bytes is not None:
                            endpoint = f"{BACKEND_INTERNAL_URL}/query/multimodal"
                            files = {"image": ("image.jpg", staged_image_bytes, "image/jpeg")}
                            data = {"question": user_input}
                            response = requests.post(endpoint, data=data, files=files, timeout=300)
                        else:
                            endpoint = f"{BACKEND_INTERNAL_URL}/query/text"
                            data = {"question": user_input}
                            response = requests.post(endpoint, data=data, timeout=300)

                        if response.status_code == 200:
                            data = response.json()
                            raw_answer = (data.get("answer") or "").strip()
                            explanation = data.get("explanation", {}) or {}

                            docs_explanation = explanation.get("documents", []) or []
                            refs_answer = extract_pdf_refs_from_answer(raw_answer)

                            all_refs = []
                            seen = set()
                            for ref in list(docs_explanation) + list(refs_answer):
                                ref = str(ref).strip()
                                if ref and ref not in seen:
                                    seen.add(ref)
                                    all_refs.append(ref)
                            st.session_state.last_refs = all_refs

                            answer = strip_references_section(raw_answer)
                            if not answer or "[no-context]" in answer.lower():
                                answer = "⚠️ No se encontró información relacionada en el grafo."

                            st.markdown(answer)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": answer, "image": None}
                            )
                        else:
                            err = f"⚠️ Error {response.status_code}: {response.text}"
                            st.error(err)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": err, "image": None}
                            )
                    except Exception as e:
                        err = f"❌ Error al conectar con el backend: {e}"
                        st.error(err)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": err, "image": None}
                        )