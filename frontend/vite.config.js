import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/execute': 'http://localhost:8000',
      '/status': 'http://localhost:8000',
      '/chat': 'http://localhost:8000',
      '/videos': 'http://localhost:8000',
      '/video-files': 'http://localhost:8000',
      '/frames': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    }
  }
})
