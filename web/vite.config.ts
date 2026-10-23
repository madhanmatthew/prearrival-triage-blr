import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The app is served by FastAPI at /app/ (see backend/main.py). In dev, API calls are proxied to uvicorn.
const API = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";
const proxied = ["/health", "/incident", "/incidents", "/resources", "/police", "/benchmark", "/prompts"];

export default defineConfig({
  base: "/app/",
  plugins: [react()],
  resolve: { alias: { "@": new URL("./src", import.meta.url).pathname } },
  server: {
    port: 5173,
    proxy: {
      ...Object.fromEntries(proxied.map((p) => [p, { target: API, changeOrigin: true }])),
      "/ws": { target: API.replace("http", "ws"), ws: true },
    },
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom", "react-router-dom"],
          map: ["leaflet", "react-leaflet"],
          charts: ["recharts"],
          motion: ["framer-motion"],
        },
      },
    },
  },
});
