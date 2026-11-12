import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // the static demo is served from a sub-path (GitHub Pages), so use relative URLs
  base: process.env.VITE_DEMO === '1' ? './' : '/',
  // maplibre-gl ships an ESM build that loads its worker via import.meta.url; pre-bundling breaks it.
  optimizeDeps: { exclude: ['maplibre-gl'] },
  build: { chunkSizeWarningLimit: 1600 }, // maplibre-gl alone is ~1 MB
  server: {
    port: 5179,
    strictPort: true,
    proxy: { '/api': 'http://127.0.0.1:8731' },
  },
})
