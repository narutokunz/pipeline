/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        charcoal: {
          50: '#f6f7f8',
          100: '#ebeeef',
          200: '#d7dadc',
          300: '#b4bac0',
          400: '#8b949d',
          500: '#6c757f',
          600: '#545c65',
          700: '#424850',
          800: '#2b2f35',
          850: '#22252a',
          900: '#1a1c20',
          950: '#111215',
        },
        teal: {
          50: '#f0f9f8',
          100: '#daf0ee',
          200: '#b7e2df',
          300: '#86cac6',
          400: '#52aba7',
          500: '#358f8b',
          600: '#277370',
          700: '#215c5a',
          800: '#1c4a48',
          900: '#183e3c',
        },
        offwhite: '#fcfcfd',
      },
      fontFamily: {
        serif: ['"Times New Roman"', 'Times', 'serif'],
        sans: ['"Times New Roman"', 'Times', 'serif'],
      }
    },
  },
  plugins: [],
}
