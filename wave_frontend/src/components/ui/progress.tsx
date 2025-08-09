import * as React from "react"
import { cn } from "../../lib/utils"

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number
  max?: number
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value = 0, max = 100, ...props }, ref) => {
    const percentage = Math.min(Math.max((value / max) * 100, 0), 100)
    
    return (
      <div
        ref={ref}
        className={cn(
          "relative h-3 w-full overflow-hidden rounded-glass glass-elev-1 border border-border-glass",
          className
        )}
        {...props}
      >
        <div 
          className="h-full bg-gradient-to-r from-accent-cyan to-accent-purple transition-all duration-500 ease-out relative overflow-hidden"
          style={{ width: `${percentage}%` }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse" />
        </div>
      </div>
    )
  }
)
Progress.displayName = "Progress"

export { Progress }