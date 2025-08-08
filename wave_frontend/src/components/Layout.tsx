import React from 'react'
import { motion } from 'framer-motion'
import { cn } from '../lib/utils'
import { useWebSocket } from '../hooks/useWebSocket'
import { Button } from './ui/button'
import { 
  LayoutDashboard, 
  Wallet, 
  TrendingUp, 
  Activity, 
  Settings,
  Wifi,
  WifiOff,
  Brain,
  Sparkles,
  Database
} from 'lucide-react'

interface LayoutProps {
  children: React.ReactNode
  currentPage: string
  onNavigate: (page: string) => void
}

const navigationItems = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'portfolio', label: 'Portfolio', icon: Wallet },
  { id: 'strategies', label: 'Strategies', icon: TrendingUp },
  { id: 'trading', label: 'Trading', icon: Activity },
  // LLM Section
  { id: 'llm-center', label: 'LLM Center', icon: Brain },
  { id: 'strategy-generator', label: 'Strategy Generator', icon: Sparkles },
  { id: 'memory-inspector', label: 'Memory Inspector', icon: Database },
  // Settings
  { id: 'settings', label: 'Settings', icon: Settings },
]

export const Layout: React.FC<LayoutProps> = ({ 
  children, 
  currentPage, 
  onNavigate 
}) => {
  const { isConnected, connectionStatus } = useWebSocket()

  return (
    <div className="flex h-screen relative">
      {/* Background with floating blobs */}
      <div className="min-h-screen w-full overflow-x-hidden text-wave-text bg-wave-main fixed inset-0">
        <div className="pointer-events-none fixed inset-0 overflow-hidden">
          <motion.div 
            className="absolute -top-24 -left-16 h-80 w-80 rounded-full bg-cyan-500/20 blur-3xl" 
            animate={{ y: [0, 20, 0] }} 
            transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }} 
          />
          <motion.div 
            className="absolute -top-10 right-0 h-96 w-96 rounded-full bg-fuchsia-500/20 blur-3xl" 
            animate={{ y: [0, -15, 0], x: [0, 10, 0] }} 
            transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }} 
          />
        </div>
      </div>
      
      {/* Sidebar */}
      <div className="w-64 wave-glass border-r border-wave-glass-border flex flex-col relative z-10">
        {/* Logo */}
        <div className="p-6 border-b border-wave-glass-border">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-br from-wave-accent-1/80 to-wave-accent-3/80 rounded-2xl flex items-center justify-center text-xl">
              🌊
            </div>
            <div>
              <h1 className="text-xl font-bold text-wave-text">Wave</h1>
              <p className="text-xs text-wave-text-secondary">Trading Platform</p>
            </div>
          </div>
        </div>

        {/* Connection Status */}
        <div className="px-6 py-3 border-b border-wave-glass-border">
          <div className="flex items-center space-x-2">
            {isConnected ? (
              <>
                <Wifi className="w-4 h-4 text-wave-accent-1" />
                <span className="text-xs text-wave-accent-1">Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-wave-bad" />
                <span className="text-xs text-wave-bad capitalize">{connectionStatus}</span>
              </>
            )}
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          {navigationItems.map((item) => {
            const Icon = item.icon
            const isActive = currentPage === item.id

            return (
              <Button
                key={item.id}
                variant={isActive ? "default" : "ghost"}
                className={cn(
                  "w-full justify-start space-x-3 h-12 transition-all duration-wave-fast",
                  isActive ? "glass-elev-2 border-wave-accent-1/20 text-wave-text shadow-glow-cyan" : "glass-hover text-wave-text-secondary hover:text-wave-text"
                )}
                onClick={() => onNavigate(item.id)}
              >
                <Icon className={cn("w-5 h-5", isActive ? "text-wave-accent-1" : "")} />
                <span>{item.label}</span>
              </Button>
            )
          })}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-wave-glass-border">
          <div className="text-xs text-wave-text-muted text-center">
            Wave v2.0 - Advanced Trading
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden relative z-10">
        {/* Header */}
        <header className="h-16 glass-elev-1 border-b border-wave-glass-border flex items-center justify-between px-6">
          <div>
            <h2 className="text-lg font-semibold text-wave-text capitalize">
              {currentPage}
            </h2>
            <p className="text-sm text-wave-text-secondary">
              {getPageDescription(currentPage)}
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Status indicators */}
            <div className="px-3 py-1 rounded-2xl text-xs font-medium bg-wave-ok/10 border border-wave-ok/20 text-wave-ok backdrop-blur-sm">
              Paper Mode
            </div>
            <div className="flex items-center space-x-2 glass-elev-1 px-3 py-1 rounded-2xl">
              <div className="w-2 h-2 bg-wave-accent-1 rounded-full animate-pulse"></div>
              <span className="text-xs text-wave-text-secondary">Live</span>
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-auto p-6">
          <div className="max-w-7xl mx-auto">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}

function getPageDescription(page: string): string {
  switch (page) {
    case 'dashboard':
      return 'Portfolio overview and market insights'
    case 'portfolio':
      return 'Your assets, positions, and performance'
    case 'strategies':
      return 'Trading strategies and algorithms'
    case 'trading':
      return 'Live trading and order management'
    case 'llm-center':
      return 'AI-powered trading intelligence'
    case 'strategy-generator':
      return 'Generate strategies with AI'
    case 'memory-inspector':
      return 'System memory and context'
    case 'settings':
      return 'Platform configuration'
    default:
      return ''
  }
}