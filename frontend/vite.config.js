import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/execute': 'http://127.0.0.1:8000',
      '/status': 'http://127.0.0.1:8000',
      '/chat': 'http://127.0.0.1:8000',
      '/videos': 'http://127.0.0.1:8000',
      '/video-files': 'http://127.0.0.1:8000',
      '/frames': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/cases': 'http://127.0.0.1:8000',
    }
  }
})
