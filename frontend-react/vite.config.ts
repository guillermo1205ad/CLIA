import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  const backend =
    env.VITE_BACKEND_INTERNAL_URL ||
    env.BACKEND_INTERNAL_URL ||
    "http://localhost:8083";

  console.log("[Vite] BACKEND =", backend);

  return {
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: 8502,
      strictPort: true,
      allowedHosts: ["grafo-nuestramemoria.pln.villena.cl"],
      proxy: {
        "/api": {
          target: backend,
          changeOrigin: true,
          // 👇 ESTA LÍNEA EVITA EL 404:
          // /api/health → http://localhost:8083/health
          // /api/query/text → http://localhost:8083/query/text
          rewrite: (path) => path.replace(/^\/api/, ""),
        },
      },
    },
    build: {
      outDir: "dist",
    },
  };
});