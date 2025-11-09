import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// IMPORTANT: set base to your repo name so GitHub Pages serves assets correctly
export default defineConfig({
  plugins: [react()],
  base: '/iluvgia/' // <- change if your repo is named differently
})
