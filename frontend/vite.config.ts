import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // maplibre-gl ships an ESM build that loads its worker via import.meta.url; pre-bundling breaks it.
  optimizeDeps: { exclude: ['maplibre-gl'] },
  server: {
    port: 5179,
    strictPort: true,
    proxy: { '/api': 'http://127.0.0.1:8731' },
  },
})
