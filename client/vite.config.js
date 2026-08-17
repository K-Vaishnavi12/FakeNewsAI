import { defineConfig } from 'vite'
import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import babel from '@rolldown/plugin-babel'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    babel({ presets: [reactCompilerPreset()] })
  ],
  server: {
    proxy: {
      // Forward /api/* straight through to Flask. The previous config had a
      // `rewrite` that STRIPPED the /api prefix, so /api/analyze reached Flask
      // as /analyze. That only worked because every route was registered
      // twice; now that routes are declared once under /api, the prefix must
      // be preserved.
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      }
    }
  }
})
