import React from 'react'
import { cn } from '../lib/utils'
import { useWebSocket } from '../hooks/useWebSocket'
import { Button } from './ui/button'
import { 
  LayoutDashboard, 
  Wallet, 
  TrendingUp, 
  Activity, 
  Settings,
  Waves,
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
      {/* Background overlay with nebula gradient */}
      <div className="absolute inset-0 bg-nebula-gradient pointer-events-none"></div>
      
      {/* Sidebar */}
      <div className="w-64 glass-elev-2 border-r border-glass flex flex-col relative z-10">
        {/* Logo */}
        <div className="p-6 border-b border-glass">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-accent-cyan rounded-glass flex items-center justify-center glow-cyan">
              <Waves className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-fg-primary text-glow-cyan">Nebula</h1>
              <p className="text-xs text-fg-secondary">Trading Platform</p>
            </div>
          </div>
        </div>

        {/* Connection Status */}
        <div className="px-6 py-3 border-b border-glass">
          <div className="flex items-center space-x-2">
            {isConnected ? (
              <>
                <Wifi className="w-4 h-4 text-accent-emerald" />
                <span className="text-xs text-accent-emerald">Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-red-400" />
                <span className="text-xs text-red-400 capitalize">{connectionStatus}</span>
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
                  "w-full justify-start space-x-3 h-12 transition-all duration-micro",
                  isActive ? "glass-elev-2 glow-cyan border-accent-cyan/20 text-fg-primary" : "glass-hover text-fg-secondary hover:text-fg-primary"
                )}
                onClick={() => onNavigate(item.id)}
              >
                <Icon className={cn("w-5 h-5", isActive ? "text-accent-cyan" : "")} />
                <span>{item.label}</span>
              </Button>
            )
          })}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-glass">
          <div className="text-xs text-fg-muted text-center">
            Nebula v2.0 - Advanced Trading
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden relative z-10">
        {/* Header */}
        <header className="h-16 glass-elev-1 border-b border-glass flex items-center justify-between px-6">
          <div>
            <h2 className="text-lg font-semibold text-fg-primary capitalize">
              {currentPage}
            </h2>
            <p className="text-sm text-fg-secondary">
              {getPageDescription(currentPage)}
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Status indicators */}
            <div className="status-healthy px-3 py-1 rounded-glass text-xs font-medium">
              Paper Mode
            </div>
            <div className="flex items-center space-x-2 glass-elev-1 px-3 py-1 rounded-glass">
              <div className="w-2 h-2 bg-accent-emerald rounded-full animate-pulse"></div>
              <span className="text-xs text-fg-secondary">Live</span>
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