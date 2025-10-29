#!/bin/bash
# ==========================================================
# 🧹 Detiene TODO el stack de Nuestra-MemorIA (GPU 2)
# ==========================================================

echo
echo "🧹 Deteniendo servicios del stack Nuestra-MemorIA..."
echo "----------------------------------------------------"

# Lista de sesiones tmux utilizadas
SESSIONS=("gptoss120b" "llava" "memoria_backend" "memoria_frontend")

# Iterar sobre las sesiones y cerrarlas si existen
for s in "${SESSIONS[@]}"; do
  if tmux has-session -t "$s" 2>/dev/null; then
    echo "   🔴 Cerrando sesión tmux: $s"
    tmux kill-session -t "$s"
  else
    echo "   ⚪ Sesión $s no encontrada (ya estaba cerrada)"
  fi
done

# Verificar procesos vLLM activos (por seguridad)
PIDS=$(pgrep -f "vllm serve")
if [ -n "$PIDS" ]; then
  echo "   ⚙️  Terminando procesos vLLM residuales..."
  kill -9 $PIDS 2>/dev/null
fi

# Verificar FastAPI o Streamlit activos (por seguridad)
BACK_PID=$(pgrep -f "main.py")
FRONT_PID=$(pgrep -f "streamlit run app.py")
if [ -n "$BACK_PID" ]; then
  echo "   ⚙️  Terminando backend FastAPI..."
  kill -9 $BACK_PID 2>/dev/null
fi
if [ -n "$FRONT_PID" ]; then
  echo "   ⚙️  Terminando frontend Streamlit..."
  kill -9 $FRONT_PID 2>/dev/null
fi

# Limpiar caché CUDA opcional
if command -v nvidia-smi &>/dev/null; then
  echo "   🧠 Liberando VRAM..."
  nvidia-smi --gpu-reset -i 2 &>/dev/null || true
fi

echo
echo "✅ Stack detenido correctamente."
echo "----------------------------------------------------"
echo "   Puedes reiniciarlo con:"
echo "   bash /home/gperalta/Nuestra-MemorIA/start_app.sh"
echo