/** Design tokens live in src/styles/index.css as CSS variables; Tailwind just exposes them. */
const rgb = (v) => `rgb(var(${v}) / <alpha-value>)`;
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["class", '[data-theme="dark"]'],
  theme: {
    extend: {
      // roomy desktop only: wide AND tall enough for floating panels over the map
      screens: { fl: { raw: "(min-width: 1280px) and (min-height: 800px)" } },
      colors: {
        base: rgb("--c-base"),
        s1: rgb("--c-s1"),
        s2: rgb("--c-s2"),
        s3: rgb("--c-s3"),
        line: rgb("--c-line"),
        fg: rgb("--c-fg"),
        mute: rgb("--c-mute"),
        faint: rgb("--c-faint"),
        brand: rgb("--c-brand"),
        onbrand: rgb("--c-onbrand"),
        sev0: rgb("--c-sev0"),
        sev1: rgb("--c-sev1"),
        sev2: rgb("--c-sev2"),
        sev3: rgb("--c-sev3"),
        sim: rgb("--c-sim"),
      },
      fontFamily: {
        sans: ['"Inter"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
        deva: ['"Noto Sans Devanagari"', '"Inter"', "sans-serif"],
        kann: ['"Noto Sans Kannada"', '"Inter"', "sans-serif"],
      },
      borderRadius: { xl: "14px", "2xl": "18px" },
      boxShadow: {
        card: "0 1px 0 rgb(255 255 255 / .03) inset, 0 8px 24px -12px rgb(0 0 0 / .5)",
        float: "0 12px 40px -12px rgb(0 0 0 / .6), 0 2px 6px rgb(0 0 0 / .25)",
        glow: "0 0 0 1px rgb(var(--c-brand) / .35), 0 0 24px -4px rgb(var(--c-brand) / .45)",
      },
      keyframes: {
        pulseRing: { "0%": { transform: "scale(.8)", opacity: ".7" }, "100%": { transform: "scale(2.4)", opacity: "0" } },
        shimmer: { "100%": { transform: "translateX(100%)" } },
      },
      animation: { pulseRing: "pulseRing 1.8s cubic-bezier(.2,.6,.3,1) infinite", shimmer: "shimmer 1.6s infinite" },
    },
  },
  plugins: [],
};
