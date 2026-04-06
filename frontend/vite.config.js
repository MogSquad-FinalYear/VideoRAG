import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/execute': 'http://localhost:8002',
      '/status': 'http://localhost:8002',
      '/chat': 'http://localhost:8002',
      '/videos': 'http://localhost:8002',
      '/video-files': 'http://localhost:8002',
      '/frames': 'http://localhost:8002',
      '/health': 'http://localhost:8002',
    }
  }
})
