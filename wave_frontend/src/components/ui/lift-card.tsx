import { motion } from "framer-motion"
import { cn } from "../../lib/utils"

interface LiftCardProps {
  children: React.ReactNode
  className?: string
  disabled?: boolean
  onClick?: () => void
}

export function LiftCard({ 
  children, 
  className = "", 
  disabled = false,
  onClick 
}: LiftCardProps) {
  return (
    <motion.div
      className={cn(
        "cursor-pointer select-none",
        disabled && "cursor-not-allowed opacity-60",
        className
      )}
      onClick={disabled ? undefined : onClick}
      whileHover={disabled ? undefined : { 
        y: -2, 
        scale: 1.01 
      }}
      whileTap={disabled ? undefined : { 
        scale: 0.997 
      }}
      transition={{ 
        type: "spring", 
        stiffness: 260, 
        damping: 20 
      }}
    >
      {children}
    </motion.div>
  )
}