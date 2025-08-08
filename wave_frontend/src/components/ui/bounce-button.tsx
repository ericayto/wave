import { motion } from "framer-motion"
import { cn } from "../../lib/utils"

interface BounceButtonProps {
  children: React.ReactNode
  className?: string
  size?: "icon" | "sm" | "md" | "lg"
  variant?: "default" | "subtle" | "ghost"
  onClick?: () => void
  disabled?: boolean
  type?: "button" | "submit" | "reset"
}

export function BounceButton({ 
  children, 
  className = "", 
  size = "md", 
  variant = "default", 
  onClick,
  disabled = false,
  type = "button"
}: BounceButtonProps) {
  const sizes = {
    icon: "h-10 w-10 grid place-items-center",
    sm: "px-3 py-1.5 text-xs",
    md: "px-4 py-2 text-sm",
    lg: "px-5 py-2.5 text-sm"
  }
  
  const variants = {
    default: `wave-glass relative overflow-hidden text-wave-text hover:bg-white/10 transition-colors`,
    subtle: `relative overflow-hidden rounded-2xl border border-wave-glass-border bg-wave-glass-bg text-wave-text-secondary hover:bg-white/10`,
    ghost: `relative overflow-hidden rounded-2xl text-wave-accent-1/90 hover:text-wave-accent-1`,
  }
  
  return (
    <motion.button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        variants[variant],
        sizes[size],
        disabled ? "opacity-60 cursor-not-allowed" : "",
        className
      )}
      whileHover={disabled ? undefined : { scale: 1.02 }}
      whileTap={disabled ? undefined : { scale: 0.985 }}
      transition={{ 
        type: "spring", 
        stiffness: 260, 
        damping: 18 
      }}
    >
      <motion.span 
        initial={{ y: 0 }} 
        animate={{ y: [0, -2, 0] }} 
        transition={{ duration: 0.28 }} 
        className="relative z-10 flex items-center gap-2"
      >
        {children}
      </motion.span>
    </motion.button>
  )
}