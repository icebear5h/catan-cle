import { fileURLToPath } from 'node:url'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

const playgroundSrc = fileURLToPath(
  new URL('../../playground/frontend/src', import.meta.url),
)
const playgroundPublic = fileURLToPath(
  new URL('../../playground/frontend/public', import.meta.url),
)
const reviewDirectory = fileURLToPath(
  new URL('../../artifacts/generated/sft/symbolic_board_fluency_review_v1/', import.meta.url),
)
const reviewFiles = ['preview.json', 'review.jsonl', 'metadata.json'] as const
const reviewPrefix = '/board-fluency-review/'

function boardFluencyReview(): Plugin {
  return {
    name: 'board-fluency-review',
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const pathname = request.url?.split('?')[0] || ''
        if (!pathname.startsWith(reviewPrefix)) return next()
        // Exact allowlist: request paths never become filesystem paths.
        const file = reviewFiles.find((name) => pathname === reviewPrefix + name)
        response.setHeader('Cache-Control', 'no-store')
        response.setHeader('X-Content-Type-Options', 'nosniff')
        if (!file) {
          response.statusCode = 404
          response.setHeader('Content-Type', 'application/json; charset=utf-8')
          response.end(JSON.stringify({ error: 'Unknown Board Fluency review artifact.' }))
          return
        }
        if (request.method !== 'GET' && request.method !== 'HEAD') {
          response.statusCode = 405
          response.setHeader('Allow', 'GET, HEAD')
          response.setHeader('Content-Type', 'application/json; charset=utf-8')
          response.end(JSON.stringify({ error: 'Board Fluency review artifacts are read-only.' }))
          return
        }
        void readFile(resolve(reviewDirectory, file)).then((contents) => {
          response.statusCode = 200
          response.setHeader('Content-Type', file.endsWith('.jsonl')
            ? 'application/x-ndjson; charset=utf-8' : 'application/json; charset=utf-8')
          response.setHeader('Content-Length', contents.length)
          if (file !== 'preview.json') response.setHeader('Content-Disposition', `attachment; filename="${file}"`)
          response.end(request.method === 'HEAD' ? undefined : contents)
        }).catch((error: NodeJS.ErrnoException) => {
          const missing = error.code === 'ENOENT'
          response.statusCode = missing ? 404 : 500
          response.setHeader('Content-Type', 'application/json; charset=utf-8')
          response.end(JSON.stringify({
            error: `${missing ? 'Missing' : 'Unable to read'} Board Fluency artifact: artifacts/generated/sft/symbolic_board_fluency_review_v1/${file}. Generate the review bundle, then retry loading.`,
          }))
        })
      })
    },
    async generateBundle() {
      for (const file of reviewFiles) {
        try {
          this.emitFile({
            type: 'asset',
            fileName: `board-fluency-review/${file}`,
            source: await readFile(resolve(reviewDirectory, file)),
          })
        } catch {
          this.error(`Cannot build Board Fluency review: unable to read artifacts/generated/sft/symbolic_board_fluency_review_v1/${file}. Generate all three review artifacts before building.`)
        }
      }
    },
  }
}

// Standalone CatanBoardBench verifier UI.
// Shares read-only view components (HexBoard, types, base styles) with the
// playground frontend via the @playground alias; owns everything bench-specific.
export default defineConfig({
  plugins: [react(), boardFluencyReview()],
  publicDir: playgroundPublic,
  resolve: {
    alias: {
      '@playground': playgroundSrc,
    },
  },
  server: {
    port: 5174,
    fs: {
      allow: ['.', playgroundSrc, playgroundPublic],
    },
  },
})
