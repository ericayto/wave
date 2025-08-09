import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "../../lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-glass px-2.5 py-0.5 text-xs font-semibold transition-all duration-micro focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "glass-elev-2 text-fg-primary border border-border-glass hover:glass-elev-3",
        secondary:
          "glass-elev-1 text-fg-secondary border border-fg-muted/20 hover:glass-elev-2",
        destructive:
          "status-critical text-red-400 hover:bg-red-400/20",
        success:
          "status-healthy text-accent-emerald hover:bg-emerald-400/20",
        warning:
          "status-warning text-yellow-400 hover:bg-yellow-400/20",
        outline:
          "glass-elev-1 border border-accent-cyan/30 text-accent-cyan hover:glass-elev-2",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }