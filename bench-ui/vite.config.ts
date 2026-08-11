import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const playgroundSrc = fileURLToPath(
  new URL('../playground/frontend/src', import.meta.url),
)

// Standalone CatanBench verifier UI.
// Shares read-only view components (HexBoard, types, base styles) with the
// playground frontend via the @playground alias; owns everything bench-specific.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@playground': playgroundSrc,
    },
  },
  server: {
    port: 5174,
    fs: {
      allow: ['.', playgroundSrc],
    },
  },
})
