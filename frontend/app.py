#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frontend de Nuestra-MemorIA 🇨🇱
Chat Multimodal con:
- Respuesta de RAGAnything real (usa /query/text o /query/multimodal)
- Referencias: visor PDF.js y descarga robusta (/download/..)
- Barra de entrada estilo ChatGPT con subida de imagen integrada
"""

import streamlit as st
import requests
import urllib.parse

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Nuestra-MemorIAnet 🇨🇱")
BACKEND_URL = "http://localhost:8083"

# CSS para que el uploader quede alineado al chat_input
st.markdown("""
<style>
[data-testid="stFileUploader"] {margin-top: -25px; padding-left: 10px;}
button[kind="secondary"] {height: 38px !important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FUNCIÓN: verificar backend
# ─────────────────────────────────────────────
def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if r.status_code == 200:
            info = r.json()
            st.sidebar.success(
                f"🟢 Backend activo\n"
                f"Modelo: {info.get('model')}\n"
                f"Storage:\n{info.get('storage')}"
            )
            return True
    except Exception as e:
        st.sidebar.error(f"❌ No se pudo conectar con el backend: {e}")
    return False

backend_ok = check_backend()

# ─────────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ─────────────────────────────────────────────
st.title("💬 Conversa con la MemorIA")

if not backend_ok:
    st.warning("⚠️ El backend no está disponible. Inícialo antes de conversar.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hola, ¿sobre qué quieres conversar?"}
        ]

    # Mostrar historial
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ─────────────────────────────────────────────
    # NUEVO BLOQUE: entrada estilo ChatGPT (texto + archivo)
    # ─────────────────────────────────────────────
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        prompt = st.chat_input("Escribe tu pregunta o suelta una imagen aquí…")
    with col2:
        uploaded_file = st.file_uploader(
            " ", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
        )

    # ─────────────────────────────────────────────
    # Lógica principal del chat
    # ─────────────────────────────────────────────
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando…"):
                try:
                    # Multimodal si hay imagen
                    if uploaded_file:
                        files = {"image": uploaded_file.getvalue()}
                        data = {"question": prompt}
                        response = requests.post(
                            f"{BACKEND_URL}/query/multimodal",
                            data=data,
                            files=files,
                            timeout=300,
                        )
                    else:
                        response = requests.post(
                            f"{BACKEND_URL}/query/text",
                            data={"question": prompt},
                            timeout=300,
                        )

                    # Manejo de respuesta
                    if response.status_code == 200:
                        data = response.json()
                        answer = (data.get("answer") or "").strip()
                        if not answer or "[no-context]" in answer.lower():
                            answer = "⚠️ No se encontró información relacionada en el grafo."

                        st.markdown(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )

                        # Mostrar referencias
                        explanation = data.get("explanation", {}) or {}
                        refs = explanation.get("documents", []) or []

                        if refs:
                            st.markdown("### 📚 Referencias")
                            for i, ref in enumerate(refs, 1):
                                filename = ref.strip()
                                encoded = urllib.parse.quote(filename)
                                viewer0 = f"{BACKEND_URL}/doc/view/0/{encoded}"
                                viewer1 = f"{BACKEND_URL}/doc/view/1/{encoded}"
                                dl0 = f"{BACKEND_URL}/download/0/{encoded}"
                                dl1 = f"{BACKEND_URL}/download/1/{encoded}"

                                st.markdown(f"**[{i}]** {filename}")
                                c1, c2, c3 = st.columns([0.34, 0.33, 0.33])
                                with c1:
                                    st.link_button("👁️ Ver (docs)", viewer0, use_container_width=True)
                                with c2:
                                    st.link_button("👁️ Ver (docs_reparar)", viewer1, use_container_width=True)
                                with c3:
                                    st.link_button("⬇️ Descargar", dl0, use_container_width=True)
                                st.caption(f"Descarga alternativa: {dl1}")

                    else:
                        st.error(f"⚠️ Error {response.status_code}: {response.text}")

                except Exception as e:
                    st.error(f"❌ Error al conectar con el backend: {e}")