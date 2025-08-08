import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Card, CardContent } from '../components/ui/card'
import { formatCurrency, formatPercent } from '../lib/utils'
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity, 
  Shield,
  Bot
} from 'lucide-react'

interface DashboardMetrics {
  portfolio_value: number
  daily_pnl: number
  daily_pnl_pct: number
  active_strategies: number
  open_orders: number
  risk_score: number
  bot_status: string
}

// Mock API call - replace with actual API
const fetchDashboardMetrics = async (): Promise<DashboardMetrics> => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 500))
  
  return {
    portfolio_value: 10000.00,
    daily_pnl: 0.00,
    daily_pnl_pct: 0.00,
    active_strategies: 0,
    open_orders: 0,
    risk_score: 10,
    bot_status: 'ready'
  }
}

export const Dashboard: React.FC = () => {
  const { data: metrics, isLoading, error } = useQuery({
    queryKey: ['dashboard-metrics'],
    queryFn: fetchDashboardMetrics,
    refetchInterval: 30000, // Refetch every 30 seconds
  })

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

      {/* Quick Actions */}
      <motion.div 
        whileHover={{ y: -2, scale: 1.01 }} 
        whileTap={{ scale: 0.997 }} 
        transition={{ type: "spring", stiffness: 260, damping: 20 }}
      >
        <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] p-6">
          <div className="flex flex-col space-y-1.5 pb-4">
            <h3 className="text-lg font-semibold leading-none tracking-tight text-zinc-200">Quick Start</h3>
            <p className="text-sm text-zinc-400">Get started with Wave</p>
          </div>
          <div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <motion.div 
                whileHover={{ y: -2, scale: 1.01 }} 
                whileTap={{ scale: 0.997 }} 
                transition={{ type: "spring", stiffness: 260, damping: 20 }}
                className="rounded-2xl border border-white/10 bg-white/5 p-4 text-center hover:bg-white/10 transition-all duration-150"
              >
                <Shield className="w-8 h-8 text-cyan-400 mx-auto mb-2" />
                <h3 className="font-medium text-zinc-200 mb-1">1. Configure Risk</h3>
                <p className="text-xs text-zinc-400">Set position limits and risk parameters</p>
              </motion.div>
              <motion.div 
                whileHover={{ y: -2, scale: 1.01 }} 
                whileTap={{ scale: 0.997 }} 
                transition={{ type: "spring", stiffness: 260, damping: 20 }}
                className="rounded-2xl border border-white/10 bg-white/5 p-4 text-center hover:bg-white/10 transition-all duration-150"
              >
                <TrendingUp className="w-8 h-8 text-violet-400 mx-auto mb-2" />
                <h3 className="font-medium text-zinc-200 mb-1">2. Choose Strategy</h3>
                <p className="text-xs text-zinc-400">Select or create a trading strategy</p>
              </motion.div>
              <motion.div 
                whileHover={{ y: -2, scale: 1.01 }} 
                whileTap={{ scale: 0.997 }} 
                transition={{ type: "spring", stiffness: 260, damping: 20 }}
                className="rounded-2xl border border-white/10 bg-white/5 p-4 text-center hover:bg-white/10 transition-all duration-150"
              >
                <Bot className="w-8 h-8 text-cyan-400 mx-auto mb-2" />
                <h3 className="font-medium text-zinc-200 mb-1">3. Start Trading</h3>
                <p className="text-xs text-zinc-400">Begin paper trading with your bot</p>
              </motion.div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}