import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { OnboardingWizard, WizardData } from '../components/OnboardingWizard'
import { formatCurrency, formatPercent } from '../lib/utils'
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity, 
  Shield,
  Bot,
  Play,
  Square,
  Settings,
  Zap
} from 'lucide-react'

interface DashboardMetrics {
  portfolio_value: number
  daily_pnl: number
  daily_pnl_pct: number
  active_strategies: number
  open_orders: number
  risk_score: number
  bot_status: string
  is_configured: boolean
  is_running: boolean
}

// Fetch dashboard metrics from backend API
const fetchDashboardMetrics = async (): Promise<DashboardMetrics> => {
  try {
    // Check configuration status from backend
    const configResponse = await fetch('http://localhost:8080/api/config/status')
    const configStatus = await configResponse.json()
    
    // For now, use mock data for portfolio metrics
    // In production, this would come from the backend too
    return {
      portfolio_value: 10000.00,
      daily_pnl: 0.00,
      daily_pnl_pct: 0.00,
      active_strategies: 0,
      open_orders: 0,
      risk_score: 10,
      bot_status: configStatus.is_configured ? 'ready' : 'not_configured',
      is_configured: configStatus.is_configured,
      is_running: false
    }
  } catch (error) {
    console.error('Failed to fetch dashboard metrics:', error)
    
    // Fallback to localStorage check
    const hasCompletedSetup = localStorage.getItem('wave_setup_completed') === 'true'
    
    return {
      portfolio_value: 10000.00,
      daily_pnl: 0.00,
      daily_pnl_pct: 0.00,
      active_strategies: 0,
      open_orders: 0,
      risk_score: 10,
      bot_status: hasCompletedSetup ? 'ready' : 'not_configured',
      is_configured: hasCompletedSetup,
      is_running: false
    }
  }
}

export const Dashboard: React.FC = () => {
  const [showOnboarding, setShowOnboarding] = useState(false)
  const [botRunning, setBotRunning] = useState(false)
  
  const { data: metrics, isLoading, error, refetch } = useQuery({
    queryKey: ['dashboard-metrics'],
    queryFn: fetchDashboardMetrics,
    refetchInterval: 30000, // Refetch every 30 seconds
  })

  const handleOnboardingComplete = async (wizardData: WizardData) => {
    console.log('Onboarding completed with data:', wizardData)
    
    try {
      // Save configuration to backend
      const response = await fetch('http://localhost:8080/api/config/save-onboarding', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(wizardData),
      })
      
      if (!response.ok) {
        throw new Error('Failed to save configuration')
      }
      
      // Also save to localStorage as backup
      localStorage.setItem('wave_setup_completed', 'true')
      localStorage.setItem('wave_config', JSON.stringify(wizardData))
      
      console.log('Configuration saved successfully')
    } catch (error) {
      console.error('Failed to save configuration:', error)
      // Still save locally as fallback
      localStorage.setItem('wave_setup_completed', 'true')
      localStorage.setItem('wave_config', JSON.stringify(wizardData))
    }
    
    // Hide onboarding and refresh metrics
    setShowOnboarding(false)
    refetch()
  }

  const handleSkipOnboarding = () => {
    setShowOnboarding(false)
  }

  const startBot = async () => {
    setBotRunning(true)
    // In real app, this would make API call to start the bot
    console.log('Starting bot...')
  }

  const stopBot = async () => {
    setBotRunning(false)
    // In real app, this would make API call to stop the bot
    console.log('Stopping bot...')
  }

  // Show onboarding wizard if not configured or explicitly requested
  if ((metrics && !metrics.is_configured) || showOnboarding) {
    return <OnboardingWizard onComplete={handleOnboardingComplete} onSkip={handleSkipOnboarding} />
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        {/* Loading skeleton */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}>
              <CardContent className="p-6">
                <div className="shimmer h-4 w-16 rounded mb-2"></div>
                <div className="shimmer h-8 w-24 rounded"></div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <Card>
          <CardContent className="p-6 text-center">
            <p className="text-red-400 mb-2">Failed to load dashboard data</p>
            <p className="text-sm text-gray-400">Please check your connection and try again</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Bot Control Header */}
      <motion.div 
        whileHover={{ y: -2, scale: 1.01 }} 
        whileTap={{ scale: 0.997 }} 
        transition={{ type: "spring", stiffness: 260, damping: 20 }}
        className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] p-6"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gradient-to-r from-cyan-400 to-violet-400 rounded-2xl flex items-center justify-center shadow-[0_0_20px_rgba(34,211,238,0.25)]">
              <Bot className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-zinc-200">Wave Trading Bot</h3>
              <p className="text-zinc-400">
                Status: <span className={`font-medium ${botRunning ? 'text-cyan-400' : 'text-amber-400'}`}>
                  {botRunning ? 'Running' : 'Ready'}
                </span>
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className={`px-3 py-1 rounded-2xl text-xs font-medium backdrop-blur-sm ${
              botRunning 
                ? 'bg-cyan-400/10 border border-cyan-400/20 text-cyan-400' 
                : 'bg-amber-400/10 border border-amber-400/20 text-amber-400'
            }`}>
              {botRunning ? 'Active' : 'Idle'}
            </div>
            
            <div className="flex space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowOnboarding(true)}
                className="text-zinc-400 hover:text-zinc-200"
              >
                <Settings className="w-4 h-4 mr-2" />
                Configure
              </Button>
              
              {botRunning ? (
                <Button
                  variant="destructive"
                  onClick={stopBot}
                  className="hover:shadow-[0_0_20px_rgba(239,68,68,0.25)]"
                >
                  <Square className="w-4 h-4 mr-2" />
                  Stop Bot
                </Button>
              ) : (
                <Button
                  onClick={startBot}
                  className="hover:shadow-[0_0_20px_rgba(52,211,153,0.25)]"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Start Bot
                </Button>
              )}
            </div>
          </div>
        </div>
        
        {!botRunning && (
          <div className="mt-4 pt-4 border-t border-white/10">
            <div className="flex items-center space-x-2 text-sm text-zinc-400">
              <Zap className="w-4 h-4 text-cyan-400" />
              <span>
                Ready to trade {JSON.parse(localStorage.getItem('wave_config') || '{}').trading?.defaultSymbols?.length || 0} pairs 
                in paper mode with {JSON.parse(localStorage.getItem('wave_config') || '{}').risk?.profile || 'moderate'} risk profile
              </span>
            </div>
          </div>
        )}
      </motion.div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Portfolio Value */}
        <motion.div 
          whileHover={{ y: -2, scale: 1.01 }} 
          whileTap={{ scale: 0.997 }} 
          transition={{ type: "spring", stiffness: 260, damping: 20 }}
          className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] hover:bg-white/10 transition-colors p-6"
        >
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <div className="text-sm font-medium text-zinc-400">
              Portfolio Value
            </div>
            <DollarSign className="h-4 w-4 text-cyan-400" />
          </div>
          <div>
            <div className="text-2xl font-bold text-zinc-200">
              {metrics ? formatCurrency(metrics.portfolio_value) : '$0.00'}
            </div>
            <p className="text-xs text-zinc-400 mt-1">
              Paper trading mode
            </p>
          </div>
        </motion.div>

        {/* Daily P&L */}
        <motion.div 
          whileHover={{ y: -2, scale: 1.01 }} 
          whileTap={{ scale: 0.997 }} 
          transition={{ type: "spring", stiffness: 260, damping: 20 }}
          className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] hover:bg-white/10 transition-colors p-6"
        >
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <div className="text-sm font-medium text-zinc-400">
              Daily P&L
            </div>
            {metrics && metrics.daily_pnl >= 0 ? (
              <TrendingUp className="h-4 w-4 text-cyan-400" />
            ) : (
              <TrendingDown className="h-4 w-4 text-rose-400" />
            )}
          </div>
          <div>
            <div className={`text-2xl font-bold ${
              metrics && metrics.daily_pnl >= 0 ? 'text-cyan-400' : 'text-rose-400'
            }`}>
              {metrics ? formatCurrency(metrics.daily_pnl) : '$0.00'}
            </div>
            <p className="text-xs text-zinc-400 mt-1">
              {metrics ? formatPercent(metrics.daily_pnl_pct) : '0.00%'}
            </p>
          </div>
        </motion.div>

        {/* Active Strategies */}
        <motion.div 
          whileHover={{ y: -2, scale: 1.01 }} 
          whileTap={{ scale: 0.997 }} 
          transition={{ type: "spring", stiffness: 260, damping: 20 }}
          className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] hover:bg-white/10 transition-colors p-6"
        >
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <div className="text-sm font-medium text-zinc-400">
              Active Strategies
            </div>
            <Activity className="h-4 w-4 text-violet-400" />
          </div>
          <div>
            <div className="text-2xl font-bold text-zinc-200">
              {metrics?.active_strategies ?? 0}
            </div>
            <p className="text-xs text-zinc-400 mt-1">
              {metrics?.open_orders ?? 0} open orders
            </p>
          </div>
        </motion.div>

        {/* Risk Score */}
        <motion.div 
          whileHover={{ y: -2, scale: 1.01 }} 
          whileTap={{ scale: 0.997 }} 
          transition={{ type: "spring", stiffness: 260, damping: 20 }}
          className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] hover:bg-white/10 transition-colors p-6"
        >
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <div className="text-sm font-medium text-zinc-400">
              Risk Score
            </div>
            <Shield className="h-4 w-4 text-cyan-400" />
          </div>
          <div>
            <div className="text-2xl font-bold text-cyan-400">
              {metrics?.risk_score ?? 0}/100
            </div>
            <p className="text-xs text-cyan-400 mt-1">
              Low Risk
            </p>
          </div>
        </motion.div>
      </div>

      {/* Bot Status Card */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <motion.div 
          className="lg:col-span-2"
          whileHover={{ y: -2, scale: 1.01 }} 
          whileTap={{ scale: 0.997 }} 
          transition={{ type: "spring", stiffness: 260, damping: 20 }}
        >
          <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] p-6">
            <div className="flex flex-col space-y-1.5 pb-4">
              <h3 className="text-lg font-semibold leading-none tracking-tight text-zinc-200">Bot Status</h3>
              <p className="text-sm text-zinc-400">Current state and recent activity</p>
            </div>
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Bot className="w-8 h-8 text-cyan-400" />
                <div>
                  <div className="font-medium text-zinc-200">Wave Trading Bot</div>
                  <div className="text-sm text-zinc-400 capitalize">
                    Status: <span className="text-cyan-400">{metrics?.bot_status ?? 'Ready'}</span>
                  </div>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-zinc-400">Mode</span>
                  <span className="text-amber-400">Paper Trading</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-zinc-400">Last Update</span>
                  <span className="text-zinc-200">Just now</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-zinc-400">Uptime</span>
                  <span className="text-zinc-200">Ready to start</span>
                </div>
              </div>

              <div className="pt-4 border-t border-white/10">
                <p className="text-sm text-zinc-400">
                  🚀 Your bot is ready! Configure your first strategy to begin paper trading.
                </p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Recent Activity */}
        <motion.div 
          whileHover={{ y: -2, scale: 1.01 }} 
          whileTap={{ scale: 0.997 }} 
          transition={{ type: "spring", stiffness: 260, damping: 20 }}
        >
          <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] p-6">
            <div className="flex flex-col space-y-1.5 pb-4">
              <h3 className="text-lg font-semibold leading-none tracking-tight text-zinc-200">Recent Activity</h3>
              <p className="text-sm text-zinc-400">Latest bot actions</p>
            </div>
            <div>
              <div className="space-y-3">
                <div className="text-center py-8 text-zinc-500">
                  <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No activity yet</p>
                  <p className="text-xs mt-1">Start a strategy to see activity here</p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Configuration Overview */}
      <motion.div 
        whileHover={{ y: -2, scale: 1.01 }} 
        whileTap={{ scale: 0.997 }} 
        transition={{ type: "spring", stiffness: 260, damping: 20 }}
      >
        <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] p-6">
          <div className="flex flex-col space-y-1.5 pb-4">
            <h3 className="text-lg font-semibold leading-none tracking-tight text-zinc-200">Current Configuration</h3>
            <p className="text-sm text-zinc-400">Your bot's active settings</p>
          </div>
          <div>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <motion.div 
                whileHover={{ y: -2, scale: 1.01 }} 
                whileTap={{ scale: 0.997 }} 
                transition={{ type: "spring", stiffness: 260, damping: 20 }}
                className="rounded-2xl border border-white/10 bg-white/5 p-4 text-center hover:bg-white/10 transition-all duration-150"
              >
                <Shield className="w-8 h-8 text-cyan-400 mx-auto mb-2" />
                <h3 className="font-medium text-zinc-200 mb-1">Exchange</h3>
                <p className="text-xs text-zinc-400">Kraken (Paper Mode)</p>
              </motion.div>
              <motion.div 
                whileHover={{ y: -2, scale: 1.01 }} 
                whileTap={{ scale: 0.997 }} 
                transition={{ type: "spring", stiffness: 260, damping: 20 }}
                className="rounded-2xl border border-white/10 bg-white/5 p-4 text-center hover:bg-white/10 transition-all duration-150"
              >
                <Activity className="w-8 h-8 text-cyan-400 mx-auto mb-2" />
                <h3 className="font-medium text-zinc-200 mb-1">AI Provider</h3>
                <p className="text-xs text-zinc-400">
                  {JSON.parse(localStorage.getItem('wave_config') || '{}').llmProvider?.provider?.toUpperCase() || 'OpenAI'}
                </p>
              </motion.div>
              <motion.div 
                whileHover={{ y: -2, scale: 1.01 }} 
                whileTap={{ scale: 0.997 }} 
                transition={{ type: "spring", stiffness: 260, damping: 20 }}
                className="rounded-2xl border border-white/10 bg-white/5 p-4 text-center hover:bg-white/10 transition-all duration-150"
              >
                <TrendingUp className="w-8 h-8 text-violet-400 mx-auto mb-2" />
                <h3 className="font-medium text-zinc-200 mb-1">Risk Profile</h3>
                <p className="text-xs text-zinc-400">
                  {JSON.parse(localStorage.getItem('wave_config') || '{}').risk?.profile?.charAt(0)?.toUpperCase() + JSON.parse(localStorage.getItem('wave_config') || '{}').risk?.profile?.slice(1) || 'Moderate'}
                </p>
              </motion.div>
              <motion.div 
                whileHover={{ y: -2, scale: 1.01 }} 
                whileTap={{ scale: 0.997 }} 
                transition={{ type: "spring", stiffness: 260, damping: 20 }}
                className="rounded-2xl border border-white/10 bg-white/5 p-4 text-center hover:bg-white/10 transition-all duration-150"
              >
                <DollarSign className="w-8 h-8 text-cyan-400 mx-auto mb-2" />
                <h3 className="font-medium text-zinc-200 mb-1">Trading Pairs</h3>
                <p className="text-xs text-zinc-400">
                  {JSON.parse(localStorage.getItem('wave_config') || '{}').trading?.defaultSymbols?.length || 0} pairs
                </p>
              </motion.div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}