import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "../../lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-glass text-sm font-medium ring-offset-background transition-all duration-micro focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 ripple",
  {
    variants: {
      variant: {
        default: "glass-elev-2 text-fg-primary hover:glass-elev-3 glow-cyan border-accent-cyan/20",
        destructive:
          "glass-elev-2 text-red-400 hover:glass-elev-3 border-red-400/20 hover:glow-emerald",
        outline:
          "glass-elev-1 border border-accent-cyan/30 text-accent-cyan hover:glass-elev-2 hover:border-accent-cyan/50",
        secondary:
          "glass-elev-2 text-fg-secondary hover:glass-elev-3 hover:text-fg-primary border-fg-secondary/20",
        ghost: "hover:glass-elev-1 hover:text-fg-primary text-fg-secondary",
        link: "text-accent-cyan underline-offset-4 hover:underline hover:text-glow-cyan",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-glass-sm px-3",
        lg: "h-11 rounded-glass-lg px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }