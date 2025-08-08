/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Dark Glassmorphism Theme
        'bg-base': 'var(--bg-base)',
        'bg-elev': {
          1: 'var(--bg-elev-1)',
          2: 'var(--bg-elev-2)',
          3: 'var(--bg-elev-3)',
        },
        'fg-primary': 'var(--fg-primary)',
        'fg-secondary': 'var(--fg-secondary)',
        'fg-muted': 'var(--fg-muted)',
        'accent-cyan': 'var(--accent-cyan)',
        'accent-purple': 'var(--accent-purple)',
        'accent-emerald': 'var(--accent-emerald)',
        'glow-cyan': 'var(--glow-cyan)',
        'glow-purple': 'var(--glow-purple)',
        'glow-emerald': 'var(--glow-emerald)',
        'border-glass': 'var(--border-glass)',
        'border-glass-strong': 'var(--border-glass-strong)',
        
        // Design system variables (legacy compatibility)
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        // Ocean theme color palette
        ocean: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
          950: '#082f49',
        },
        wave: {
          50: '#ecfeff',
          100: '#cffafe',
          200: '#a5f3fc',
          300: '#67e8f9',
          400: '#22d3ee',
          500: '#06b6d4',
          600: '#0891b2',
          700: '#0e7490',
          800: '#155e75',
          900: '#164e63',
          950: '#083344',
        },
        deep: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
          950: '#020617',
        }
      },
      backgroundImage: {
        'glass-gradient': 'var(--gradient-glass)',
        'bg-gradient': 'var(--gradient-bg)',
        'ocean-gradient': 'linear-gradient(135deg, #0ea5e9 0%, #06b6d4 25%, #0891b2 50%, #0e7490 75%, #164e63 100%)',
        'wave-gradient': 'linear-gradient(135deg, #22d3ee 0%, #06b6d4 50%, #0891b2 100%)',
        'deep-gradient': 'linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #020617 100%)',
        'nebula-gradient': 'radial-gradient(ellipse at top left, rgba(139, 92, 246, 0.15) 0%, rgba(6, 182, 212, 0.1) 25%, rgba(16, 185, 129, 0.05) 50%, transparent 100%)',
      },
      
      backdropBlur: {
        'glass-low': 'var(--blur-low)',
        'glass-med': 'var(--blur-med)', 
        'glass-high': 'var(--blur-high)',
      },
      
      boxShadow: {
        'glass': 'var(--shadow-glass)',
        'elevated': 'var(--shadow-elevated)',
        'glow-cyan': '0 0 20px var(--glow-cyan)',
        'glow-purple': '0 0 20px var(--glow-purple)',
        'glow-emerald': '0 0 20px var(--glow-emerald)',
      },
      
      transitionTimingFunction: {
        'glass': 'var(--ease-standard)',
        'emphasized': 'var(--ease-emphasized)',
      },
      
      transitionDuration: {
        'micro': 'var(--dur-micro)',
        'overlay': 'var(--dur-overlay)',
      },
      animation: {
        'wave': 'wave 3s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'ripple': 'ripple 2s ease-out infinite',
        'glass-shimmer': 'glass-shimmer 2s infinite',
        'glow-pulse': 'glow-pulse 2s ease-in-out infinite alternate',
      },
      keyframes: {
        wave: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        ripple: {
          '0%': { transform: 'scale(0)', opacity: 1 },
          '100%': { transform: 'scale(4)', opacity: 0 },
        },
        'glass-shimmer': {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
        'glow-pulse': {
          '0%': { boxShadow: '0 0 20px var(--glow-cyan)' },
          '100%': { boxShadow: '0 0 40px var(--glow-cyan), 0 0 60px var(--glow-cyan)' },
        },
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'Consolas', 'monospace'],
      },
      
      borderRadius: {
        'glass': '16px',
        'glass-sm': '12px', 
        'glass-lg': '24px',
      },
    },
  },
  plugins: [
    require('tailwindcss-animate'),
  ],
}