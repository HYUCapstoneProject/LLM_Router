import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 로컬 게이트웨이 주소. 다른 포트로 띄워 붙여볼 수 있게 환경변수로 뺀다.
//   VITE_PROXY_TARGET=http://127.0.0.1:8099 npm run dev
const proxyTarget = process.env.VITE_PROXY_TARGET ?? 'http://localhost:8080'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: proxyTarget,
        changeOrigin: true,
      },
    },
  },
})
