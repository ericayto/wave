import * as React from "react"
import { cn } from "../../lib/utils"

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-10 w-full rounded-glass glass-elev-1 border border-border-glass bg-transparent px-3 py-2 text-sm text-fg-primary placeholder:text-fg-muted focus:outline-none focus:ring-2 focus:ring-accent-cyan focus:border-accent-cyan/50 disabled:cursor-not-allowed disabled:opacity-50 transition-all duration-micro",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

export { Input }