import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ['spark-e26c', '.ts.net'],
    proxy: {
      '/layers': 'http://localhost:8001',
      '/status': 'http://localhost:8001',
      '/generate': 'http://localhost:8001',
      '/edit': 'http://localhost:8001',
      '/extract-material': 'http://localhost:8001',
      '/image-to-pbr': 'http://localhost:8001',
      '/adjust-height': 'http://localhost:8001',
      '/composite-pbr': 'http://localhost:8001',
      '/export': 'http://localhost:8001',
      '/download-all': 'http://localhost:8001',
      '/library': 'http://localhost:8001',
      '/library-files': 'http://localhost:8001',
      '/outputs': 'http://localhost:8001',
    },
  },
})
