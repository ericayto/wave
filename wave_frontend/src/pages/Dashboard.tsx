import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
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
        <Card className="glow-hover hover:glow-cyan">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-fg-secondary">
              Portfolio Value
            </CardTitle>
            <DollarSign className="h-4 w-4 text-accent-cyan" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-fg-primary">
              {metrics ? formatCurrency(metrics.portfolio_value) : '$0.00'}
            </div>
            <p className="text-xs text-fg-secondary mt-1">
              Paper trading mode
            </p>
          </CardContent>
        </Card>

        {/* Daily P&L */}
        <Card className="glow-hover hover:glow-cyan">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-fg-secondary">
              Daily P&L
            </CardTitle>
            {metrics && metrics.daily_pnl >= 0 ? (
              <TrendingUp className="h-4 w-4 text-accent-cyan" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-400" />
            )}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${
              metrics && metrics.daily_pnl >= 0 ? 'text-accent-cyan' : 'text-red-400'
            }`}>
              {metrics ? formatCurrency(metrics.daily_pnl) : '$0.00'}
            </div>
            <p className="text-xs text-fg-secondary mt-1">
              {metrics ? formatPercent(metrics.daily_pnl_pct) : '0.00%'}
            </p>
          </CardContent>
        </Card>

        {/* Active Strategies */}
        <Card className="glow-hover hover:glow-purple">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-fg-secondary">
              Active Strategies
            </CardTitle>
            <Activity className="h-4 w-4 text-accent-purple" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-fg-primary">
              {metrics?.active_strategies ?? 0}
            </div>
            <p className="text-xs text-fg-secondary mt-1">
              {metrics?.open_orders ?? 0} open orders
            </p>
          </CardContent>
        </Card>

        {/* Risk Score */}
        <Card className="glow-hover hover:glow-cyan">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-fg-secondary">
              Risk Score
            </CardTitle>
            <Shield className="h-4 w-4 text-accent-cyan" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-accent-cyan">
              {metrics?.risk_score ?? 0}/100
            </div>
            <p className="text-xs text-accent-cyan mt-1">
              Low Risk
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Bot Status Card */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Bot Status</CardTitle>
            <CardDescription>Current state and recent activity</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center space-x-3">
              <Bot className="w-8 h-8 text-accent-cyan" />
              <div>
                <div className="font-medium text-fg-primary">Wave Trading Bot</div>
                <div className="text-sm text-fg-secondary capitalize">
                  Status: <span className="text-accent-cyan">{metrics?.bot_status ?? 'Ready'}</span>
                </div>
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-fg-secondary">Mode</span>
                <span className="text-yellow-400">Paper Trading</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-fg-secondary">Last Update</span>
                <span className="text-fg-primary">Just now</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-fg-secondary">Uptime</span>
                <span className="text-fg-primary">Ready to start</span>
              </div>
            </div>

            <div className="pt-4 border-t border-glass">
              <p className="text-sm text-fg-secondary">
                🚀 Your bot is ready! Configure your first strategy to begin paper trading.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>Latest bot actions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="text-center py-8 text-fg-muted">
                <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No activity yet</p>
                <p className="text-xs mt-1">Start a strategy to see activity here</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Start</CardTitle>
          <CardDescription>Get started with Wave</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="glass-elev-1 p-4 rounded-glass text-center hover:glass-elev-2 transition-all duration-micro">
              <Shield className="w-8 h-8 text-accent-cyan mx-auto mb-2" />
              <h3 className="font-medium text-fg-primary mb-1">1. Configure Risk</h3>
              <p className="text-xs text-fg-secondary">Set position limits and risk parameters</p>
            </div>
            <div className="glass-elev-1 p-4 rounded-glass text-center hover:glass-elev-2 transition-all duration-micro">
              <TrendingUp className="w-8 h-8 text-accent-purple mx-auto mb-2" />
              <h3 className="font-medium text-fg-primary mb-1">2. Choose Strategy</h3>
              <p className="text-xs text-fg-secondary">Select or create a trading strategy</p>
            </div>
            <div className="glass-elev-1 p-4 rounded-glass text-center hover:glass-elev-2 transition-all duration-micro">
              <Bot className="w-8 h-8 text-accent-cyan mx-auto mb-2" />
              <h3 className="font-medium text-fg-primary mb-1">3. Start Trading</h3>
              <p className="text-xs text-fg-secondary">Begin paper trading with your bot</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}