/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      colors: {
        'brand-cyan': '#00ffff',
        'brand-dark': '#0a101f',
        'brand-light': '#e6f9ff',
      },
    },
  },
  plugins: [],
}

