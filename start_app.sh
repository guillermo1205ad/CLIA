#!/bin/bash
# ===================================================================
# 🚀 Stack CLIA en GPU específica (ÍNDICE + venv + tmux)
#    - GPU por índice (CUDA_DEVICE_ORDER=PCI_BUS_ID)
#    - venv: ~/multimodal_graph/multimodal_graph_env
#    - vLLM vía "python -m ..." (sin depender del binario)
#    - Backend FastAPI: ~/CLIA/backend/main.py
#    - Frontend React (Vite): ~/CLIA/frontend-react
# ===================================================================
set -euo pipefail

# -----------------------------
# 0) Parámetros / Paths base
# -----------------------------
DEFAULT_GPU_ID=2
GPU_ID="${1:-$DEFAULT_GPU_ID}"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "❌ GPU inválida: $GPU_ID"; exit 1; }

HOME_DIR="$HOME"
LOGDIR="$HOME_DIR/logs_memoria"
mkdir -p "$LOGDIR"

APP_BACKEND_DIR="$HOME_DIR/CLIA/backend"
APP_FRONTEND_DIR="$HOME_DIR/CLIA/frontend-react"

# venv (el mismo que usas para todo el grafo)
VENV_DIR="$HOME_DIR/multimodal_graph/multimodal_graph_env"
VENV_ACT="$VENV_DIR/bin/activate"
if [ ! -f "$VENV_ACT" ]; then
  echo "❌ No existe el venv en: $VENV_ACT"
  exit 1
fi
VENV_PY="$VENV_DIR/bin/python"

# Model roots candidatos
MODEL_ROOTS=(
  "$HOME_DIR/models"
  "$HOME_DIR/multimodal_graph/models"
  "/home/gperalta/models"
)

GPT20B_NAME="gpt-oss-20b"

# -----------------------------
# 1) Entorno GPU + Offline HF
# -----------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# URLs backend (las heredan backend y frontend)
export BACKEND_INTERNAL_URL="http://localhost:8083"
export BACKEND_BROWSER_URL="https://grafo-nuestramemoria.pln.villena.cl"
export BACKEND_URL="$BACKEND_INTERNAL_URL"

echo
echo "🚀 Iniciando stack CLIA en GPU FÍSICA $GPU_ID…"
echo "📂 Logs: $LOGDIR"
echo "🌐 BACKEND_INTERNAL_URL = $BACKEND_INTERNAL_URL"
echo "🌐 BACKEND_BROWSER_URL  = $BACKEND_BROWSER_URL"
echo "----------------------------------------------------"

# -----------------------------
# 2) Utilidades
# -----------------------------
pick_model_dir () {
  local model="$1"; shift
  local root
  for root in "$@"; do
    if [ -d "$root/$model" ]; then
      echo "$root/$model"
      return 0
    fi
  done
  return 1
}

wait_http_ready () {
  # Uso: wait_http_ready "Nombre" "http://127.0.0.1:PUERTO/v1/models" "tmux_session" "ruta_log" "timeout_s"
  local name="$1" url="$2" session="$3" log="$4" timeout="${5:-900}"
  echo "⏳ Esperando a que $name responda en $url (timeout ${timeout}s)…"
  local start now; start=$(date +%s)
  while true; do
    if curl -s -o /dev/null -m 2 -w "%{http_code}" "$url" | grep -qE '200|404'; then
      echo "✅ $name está respondiendo en $url"
      return 0
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "❌ La sesión $session terminó. Últimas líneas de $log:"
      tail -n 200 "$log" || true
      return 1
    fi
    now=$(date +%s)
    if (( now - start > timeout )); then
      echo "⏰ Timeout esperando $name. Últimas líneas de $log:"
      tail -n 200 "$log" || true
      return 1
    fi
    sleep 3
  done
}

# -----------------------------
# 3) Autodetectar ruta del modelo
# -----------------------------
GPT20B_DIR="$(pick_model_dir "$GPT20B_NAME" "${MODEL_ROOTS[@]}")" || {
  echo "❌ No encuentro el modelo $GPT20B_NAME en: ${MODEL_ROOTS[*]}"
  echo "   Crea un symlink o ajusta MODEL_ROOTS en este script."
  exit 1
}

# -----------------------------
# 4) Servicios: tmux
# -----------------------------
launch_tmux_script () {
  local session="$1"
  local log="$2"
  local body="$3"

  tmux new-session -d -s "$session" "bash -lc '
set -e
echo \"========== [$session] bootstrap ==========\" >> $log
echo \"CWD before: \$(pwd)\" >> $log
echo \"ENV CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\" >> $log
echo \"ENV BACKEND_INTERNAL_URL=\$BACKEND_INTERNAL_URL\" >> $log
echo \"ENV BACKEND_BROWSER_URL=\$BACKEND_BROWSER_URL\" >> $log
if [ -f \"$VENV_ACT\" ]; then
  source \"$VENV_ACT\"
  echo \"VENV OK: \$VIRTUAL_ENV\" >> $log
else
  echo \"❌ VENV NOT FOUND: $VENV_ACT\" >> $log
  exit 11
fi
which python >> $log || true
python -V   >> $log || true

{
  echo \"========== [$session] launching ==========\"
  $body
} >> $log 2>&1
'"

  echo "✅ Sesión $session lanzada."
}

# 4.1 GPT-OSS-20B (vLLM)
echo "🧠 Iniciando GPT-OSS-20B (vLLM) en GPU $GPU_ID…"
GPT_LOG="$LOGDIR/gptoss20b.log"
: > "$GPT_LOG"
GPT_BODY="
cd $HOME_DIR/multimodal_graph
printf \"[gptoss20b] CWD: %s\n\" \$(pwd)
printf \"[gptoss20b] ENV CUDA_VISIBLE_DEVICES=%s\n\" \$CUDA_VISIBLE_DEVICES
python -m vllm.entrypoints.openai.api_server \
  --model \"$GPT20B_DIR\" \
  --host 0.0.0.0 --port 8010 \
  --served-model-name gpt-oss-20b \
  --tensor-parallel-size 1 \
  --max-model-len 32000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --enforce-eager
"
launch_tmux_script "gptoss20b" "$GPT_LOG" "$GPT_BODY"
wait_http_ready "GPT-OSS-20B" "http://127.0.0.1:8010/v1/models" "gptoss20b" "$GPT_LOG" 900

# 4.2 Backend FastAPI
echo "🧩 Iniciando Backend FastAPI…"
BACKEND_LOG="$LOGDIR/clia_backend.log"
BACKEND_BODY="
cd $APP_BACKEND_DIR
python3 main.py
"
launch_tmux_script "clia_backend" "$BACKEND_LOG" "$BACKEND_BODY"
echo "✅ Backend lanzado (puerto 8083)."

# 4.3 Frontend React (Vite)
echo "🌐 Iniciando Frontend React…"
FRONTEND_LOG="$LOGDIR/clia_frontend.log"
FRONTEND_BODY="
cd $APP_FRONTEND_DIR
if [ ! -d node_modules ]; then
  echo \"[frontend] Instalando dependencias (npm install)…\"
  npm install
fi
npm run dev -- --host 0.0.0.0 --port 8502
"
launch_tmux_script "clia_frontend" "$FRONTEND_LOG" "$FRONTEND_BODY"
echo "✅ Frontend React lanzado (puerto 8502)."

# -----------------------------
# 5) Resumen
# -----------------------------
echo "----------------------------------------------------"
echo "✅ Stack CLIA levantado en GPU FÍSICA $GPU_ID"
echo "   🧠 GPT-OSS-20B ➜ http://localhost:8010/v1"
echo "   🧩 Backend      ➜ http://localhost:8083"
echo "   🌐 Frontend     ➜ http://localhost:8502"
echo
echo "🌐 BACKEND_INTERNAL_URL = $BACKEND_INTERNAL_URL"
echo "🌐 BACKEND_BROWSER_URL  = $BACKEND_BROWSER_URL"
echo
echo "🔎 Ver GPU:  nvidia-smi -i $GPU_ID"
echo "🧪 Probar LLM:         curl -s http://127.0.0.1:8010/v1/models | head"
echo "🧪 Probar health back: curl -s http://127.0.0.1:8083/health"
echo
echo "💤 Para detener sesiones:"
echo "    tmux kill-session -t gptoss20b"
echo "    tmux kill-session -t clia_backend"
echo "    tmux kill-session -t clia_frontend"
echo "----------------------------------------------------"