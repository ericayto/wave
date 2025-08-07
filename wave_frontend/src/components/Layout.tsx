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
    <div className="flex h-screen bg-deep-900">
      {/* Sidebar */}
      <div className="w-64 glass border-r border-white/10 flex flex-col">
        {/* Logo */}
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-ocean-gradient rounded-lg flex items-center justify-center">
              <Waves className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Wave</h1>
              <p className="text-xs text-gray-400">Trading Bot</p>
            </div>
          </div>
        </div>

        {/* Connection Status */}
        <div className="px-6 py-3 border-b border-white/10">
          <div className="flex items-center space-x-2">
            {isConnected ? (
              <>
                <Wifi className="w-4 h-4 text-green-400" />
                <span className="text-xs text-green-400">Connected</span>
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
                  "w-full justify-start space-x-3 h-12",
                  isActive && "glow"
                )}
                onClick={() => onNavigate(item.id)}
              >
                <Icon className="w-5 h-5" />
                <span>{item.label}</span>
              </Button>
            )
          })}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-white/10">
          <div className="text-xs text-gray-500 text-center">
            Wave v0.2.0 M2 - Paper Mode + LLM
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-16 glass border-b border-white/10 flex items-center justify-between px-6">
          <div>
            <h2 className="text-lg font-semibold text-white capitalize">
              {currentPage}
            </h2>
            <p className="text-sm text-gray-400">
              {getPageDescription(currentPage)}
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Status indicators can go here */}
            <div className="status-healthy px-3 py-1 rounded-full text-xs font-medium">
              Paper Mode
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
      return 'Overview of your trading bot performance'
    case 'portfolio':
      return 'Your balances, positions, and P&L'
    case 'strategies':
      return 'Manage and configure trading strategies'
    case 'trading':
      return 'Orders, trades, and execution history'
    case 'llm-center':
      return 'LLM control, planning, and query interface'
    case 'strategy-generator':
      return 'Generate strategies from natural language'
    case 'memory-inspector':
      return 'Memory, context, and RAG management'
    case 'settings':
      return 'Configuration and system settings'
    default:
      return ''
  }
}