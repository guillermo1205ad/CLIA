#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frontend de CLIA 🇨🇱
"""

import streamlit as st
import requests
import urllib.parse
import os
import base64
import re

# ─────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────

def extract_pdf_refs_from_answer(answer: str) -> list[str]:
    """
    Busca líneas del tipo:
      - [1] 1071_foia_...pdf
    dentro del texto de la respuesta y devuelve la lista de nombres .pdf.
    """
    if not answer:
        return []
    pattern = r"\[\d+\]\s+(.+?\.pdf)"
    return re.findall(pattern, answer)


def strip_references_section(answer: str) -> str:
    """
    Elimina la sección 'References' o 'Referencias' (en inglés o español,
    con o sin ###, -, *, etc.) y todo lo que viene después.
    Así mostramos las referencias solo en la sección de abajo.
    """
    if not answer:
        return answer

    lines = answer.splitlines()
    cut_idx = None

    for i, line in enumerate(lines):
        # Quitar markdown simple (#, -, *, espacios) al inicio
        norm = re.sub(r'^[#\-\*\s]+', '', line).strip().lower()
        if norm.startswith("references") or norm.startswith("referencias"):
            cut_idx = i
            break

    if cut_idx is None:
        return answer

    kept = lines[:cut_idx]
    # Quitar líneas en blanco al final
    while kept and not kept[-1].strip():
        kept.pop()

    return "\n".join(kept)


def render_references(BACKEND_INTERNAL_URL: str):
    """
    Renderiza las referencias almacenadas en st.session_state.last_refs
    como una única sección "Referencias", donde cada texto
    "[n] nombre.pdf" ES el botón de descarga.
    """
    all_refs = st.session_state.get("last_refs", [])
    if not all_refs:
        return

    st.markdown("---")
    st.markdown("#### Referencias")

    max_refs_to_show = st.session_state.get("max_refs_to_show", 10)
    refs_to_show = all_refs[:max_refs_to_show]
    hidden = max(0, len(all_refs) - len(refs_to_show))

    for i, filename in enumerate(refs_to_show, 1):
        encoded = urllib.parse.quote(filename)
        download_internal_url = f"{BACKEND_INTERNAL_URL}/download/0/{encoded}"

        try:
            resp = requests.get(download_internal_url, timeout=60)
            if resp.status_code == 200:
                st.download_button(
                    label=f"[{i}] {filename}",
                    data=resp.content,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"dl_{i}_{filename}",
                )
            else:
                st.error(
                    f"No se pudo preparar la descarga de [{i}] "
                    f"(HTTP {resp.status_code})."
                )
        except Exception as e:
            st.error(f"Error al preparar la descarga de [{i}]: {e}")

    if hidden > 0:
        st.caption(f"… y {hidden} referencias adicionales no mostradas.")


# ─────────────────────────────────────────────
# CONFIG Y ESTILO
# ─────────────────────────────────────────────

st.set_page_config(layout="centered", page_title="Antígona")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8083")
BACKEND_INTERNAL_URL = os.getenv("BACKEND_INTERNAL_URL", BACKEND_URL)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    [data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    [data-testid="stExpander"] summary {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# AVATARES
# ─────────────────────────────────────────────

ASSISTANT_AVATAR = "static/icono.png"   # Sherlock
USER_AVATAR = "static/user.png"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("Configuración")

    try:
        r = requests.get(f"{BACKEND_INTERNAL_URL}/health", timeout=5)
        if r.status_code == 200:
            info = r.json()
            st.success(
                f"🟢 Backend activo\n"
                f"Modelo: {info.get('model')}\n"
                f"Storage:\n{info.get('storage')}"
            )
            backend_ok = True
        else:
            st.error(f"❌ Backend con error: {r.status_code}")
            backend_ok = False
    except Exception as e:
        st.error("❌ Backend desconectado")
        st.caption(str(e))
        backend_ok = False

    st.divider()

    st.header("Carga Multimodal")
    uploaded_file = st.file_uploader(
        "Sube una imagen para analizar",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.session_state.staged_image = uploaded_file.getvalue()
        st.image(st.session_state.staged_image, "Imagen preparada:")

    if "staged_image" in st.session_state and not uploaded_file:
        st.image(st.session_state.staged_image, "Imagen preparada:")
        if st.button("Quitar Imagen"):
            del st.session_state.staged_image
            st.rerun()

    st.divider()
    st.subheader("Referencias a mostrar")
    st.slider(
        "Máximo de referencias",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        key="max_refs_to_show",
        help="Controla cuántas referencias se muestran por respuesta."
    )


# ─────────────────────────────────────────────
# TÍTULO CON ICONO ALINEADO (FLEXBOX)
# ─────────────────────────────────────────────

icon_path = "static/icono.png"
icon_b64 = ""
try:
    with open(icon_path, "rb") as f:
        icon_b64 = base64.b64encode(f.read()).decode("utf-8")
except Exception:
    pass

if icon_b64:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
            <img src="data:image/png;base64,{icon_b64}" width="60" style="border-radius:15px;" />
            <h1 style="margin:0;">Antígona</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("Antígona")


# ─────────────────────────────────────────────
# INTERFAZ DE CHAT
# ─────────────────────────────────────────────

if not backend_ok:
    st.warning("⚠️ El backend no está disponible. No se puede iniciar el chat.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¿Qué vamos a explorar hoy?"}
        ]
    if "last_refs" not in st.session_state:
        st.session_state.last_refs = []

    # Historial
    for msg in st.session_state.messages:
        avatar = ASSISTANT_AVATAR if msg["role"] == "assistant" else USER_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Entrada de usuario
    prompt = st.chat_input("Pregunta lo que quieras")

    if prompt:
        staged_image_bytes = st.session_state.pop("staged_image", None)

        with st.chat_message("user", avatar=USER_AVATAR):
            if staged_image_bytes:
                st.image(staged_image_bytes, width=200)
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        # Respuesta asistente
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Pensando…"):
                try:
                    # Llamada al backend
                    if staged_image_bytes:
                        endpoint = f"{BACKEND_INTERNAL_URL}/query/multimodal"
                        files = {"image": ("image.jpg", staged_image_bytes, "image/jpeg")}
                        data = {"question": prompt}
                        response = requests.post(endpoint, data=data, files=files, timeout=300)
                    else:
                        endpoint = f"{BACKEND_INTERNAL_URL}/query/text"
                        data = {"question": prompt}
                        response = requests.post(endpoint, data=data, timeout=300)

                    if response.status_code == 200:
                        data = response.json()
                        raw_answer = (data.get("answer") or "").strip()

                        # 1) Sacamos referencias
                        explanation = data.get("explanation", {}) or {}
                        refs_explanation = explanation.get("documents", []) or []
                        refs_answer = extract_pdf_refs_from_answer(raw_answer)

                        all_refs = []
                        seen = set()
                        for ref in list(refs_explanation) + list(refs_answer):
                            ref = str(ref).strip()
                            if ref and ref not in seen:
                                seen.add(ref)
                                all_refs.append(ref)

                        st.session_state.last_refs = all_refs

                        # 2) Quitamos la sección "References/Referencias" del texto
                        answer = strip_references_section(raw_answer)
                        if not answer or "[no-context]" in answer.lower():
                            answer = "⚠️ No se encontró información relacionada en el grafo."

                        st.markdown(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )

                        # 3) Mostrar referencias abajo
                        render_references(BACKEND_INTERNAL_URL)

                    else:
                        err = f"⚠️ Error {response.status_code}: {response.text}"
                        st.error(err)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": err}
                        )

                except Exception as e:
                    err = f"❌ Error al conectar con el backend: {e}"
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )

    else:
        # Sin nuevo prompt (por ejemplo, tras hacer clic en un botón de descarga):
        # solo volvemos a dibujar las referencias de la última respuesta.
        render_references(BACKEND_INTERNAL_URL)