import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { formatCurrency, formatPercent } from '../lib/utils'
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  Activity,
  Calendar,
  Target,
  Brain,
  AlertTriangle,
  CheckCircle,
  Clock,
  DollarSign,
  Percent,
  Shield,
  Zap
} from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select'

interface PerformanceMetrics {
  total_return: number
  annualized_return: number
  sharpe_ratio: number
  sortino_ratio: number
  calmar_ratio: number
  max_drawdown: number
  max_drawdown_duration: number
  volatility: number
  win_rate: number
  profit_factor: number
  current_streak: number
  max_winning_streak: number
}

interface RegimeData {
  current_regime: 'trending' | 'mean_reverting' | 'volatile' | 'consolidating'
  regime_confidence: number
  regime_duration: number
  regime_history: Array<{
    regime: string
    start_date: string
    end_date: string
    performance: number
  }>
}

interface CorrelationData {
  strategy_correlations: Array<{
    strategy1: string
    strategy2: string
    correlation: number
  }>
  asset_correlations: Array<{
    asset1: string
    asset2: string
    correlation: number
  }>
}

interface AdvancedAnalyticsData {
  performance_metrics: PerformanceMetrics
  regime_data: RegimeData
  correlation_data: CorrelationData
  chart_data: {
    daily_returns: Array<{ date: string, return: number, cumulative: number }>
    drawdown_chart: Array<{ date: string, drawdown: number }>
    rolling_sharpe: Array<{ date: string, sharpe: number }>
  }
}

// Mock API call for advanced analytics
const fetchAdvancedAnalytics = async (timeframe: string): Promise<AdvancedAnalyticsData> => {
  await new Promise(resolve => setTimeout(resolve, 1000))
  
  // Generate mock data
  const mockData: AdvancedAnalyticsData = {
    performance_metrics: {
      total_return: 0.1247,
      annualized_return: 0.1891,
      sharpe_ratio: 1.34,
      sortino_ratio: 1.87,
      calmar_ratio: 2.15,
      max_drawdown: -0.0587,
      max_drawdown_duration: 12,
      volatility: 0.1423,
      win_rate: 0.642,
      profit_factor: 1.87,
      current_streak: 3,
      max_winning_streak: 8
    },
    regime_data: {
      current_regime: 'trending',
      regime_confidence: 0.84,
      regime_duration: 7,
      regime_history: [
        { regime: 'trending', start_date: '2024-01-15', end_date: '2024-01-22', performance: 0.045 },
        { regime: 'volatile', start_date: '2024-01-08', end_date: '2024-01-14', performance: -0.012 },
        { regime: 'mean_reverting', start_date: '2024-01-01', end_date: '2024-01-07', performance: 0.021 }
      ]
    },
    correlation_data: {
      strategy_correlations: [
        { strategy1: 'Momentum', strategy2: 'Mean Reversion', correlation: -0.23 },
        { strategy1: 'Momentum', strategy2: 'Breakout', correlation: 0.67 },
        { strategy1: 'Mean Reversion', strategy2: 'Arbitrage', correlation: 0.15 }
      ],
      asset_correlations: [
        { asset1: 'BTC', asset2: 'ETH', correlation: 0.82 },
        { asset1: 'BTC', asset2: 'ADA', correlation: 0.71 },
        { asset1: 'ETH', asset2: 'ADA', correlation: 0.89 }
      ]
    },
    chart_data: {
      daily_returns: Array.from({ length: 30 }, (_, i) => ({
        date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        return: (Math.random() - 0.5) * 0.04,
        cumulative: Math.random() * 0.15 - 0.02
      })),
      drawdown_chart: Array.from({ length: 30 }, (_, i) => ({
        date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        drawdown: -Math.random() * 0.08
      })),
      rolling_sharpe: Array.from({ length: 30 }, (_, i) => ({
        date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        sharpe: Math.random() * 2.5 + 0.5
      }))
    }
  }
  
  return mockData
}

export const AdvancedAnalytics: React.FC = () => {
  const [timeframe, setTimeframe] = useState('30d')
  
  const { data: analytics, isLoading, error, refetch } = useQuery({
    queryKey: ['advanced-analytics', timeframe],
    queryFn: () => fetchAdvancedAnalytics(timeframe),
    refetchInterval: 60000, // Refetch every minute
  })

  const getRegimeColor = (regime: string) => {
    switch (regime) {
      case 'trending': return 'text-green-400 bg-green-400/10'
      case 'volatile': return 'text-yellow-400 bg-yellow-400/10'
      case 'mean_reverting': return 'text-blue-400 bg-blue-400/10'
      case 'consolidating': return 'text-purple-400 bg-purple-400/10'
      default: return 'text-gray-400 bg-gray-400/10'
    }
  }

  const getRegimeIcon = (regime: string) => {
    switch (regime) {
      case 'trending': return <TrendingUp className="w-4 h-4" />
      case 'volatile': return <Zap className="w-4 h-4" />
      case 'mean_reverting': return <Activity className="w-4 h-4" />
      case 'consolidating': return <Target className="w-4 h-4" />
      default: return <Brain className="w-4 h-4" />
    }
  }

  const getCorrelationColor = (correlation: number) => {
    if (correlation >= 0.7) return 'text-red-400 bg-red-400/10'
    if (correlation >= 0.3) return 'text-yellow-400 bg-yellow-400/10'
    if (correlation >= -0.3) return 'text-gray-400 bg-gray-400/10'
    return 'text-green-400 bg-green-400/10'
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white">Advanced Analytics</h1>
            <p className="text-gray-400 mt-2">Deep performance insights and market regime analysis</p>
          </div>
          <div className="shimmer h-10 w-32 rounded"></div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {Array.from({ length: 8 }).map((_, i) => (
            <Card key={i}>
              <CardContent className="p-6">
                <div className="shimmer h-4 w-20 rounded mb-2"></div>
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
            <AlertTriangle className="w-8 h-8 text-red-400 mx-auto mb-2" />
            <p className="text-red-400 mb-2">Failed to load analytics data</p>
            <p className="text-sm text-gray-400 mb-4">Please check your connection and try again</p>
            <Button onClick={() => refetch()} variant="outline" size="sm">
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Advanced Analytics</h1>
          <p className="text-gray-400 mt-2">Deep performance insights and market regime analysis</p>
        </div>
        <div className="flex items-center space-x-4">
          <Select value={timeframe} onValueChange={setTimeframe}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="7d">7 Days</SelectItem>
              <SelectItem value="30d">30 Days</SelectItem>
              <SelectItem value="90d">90 Days</SelectItem>
              <SelectItem value="1y">1 Year</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Performance Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Return */}
        <Card className="glow-hover">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">
              Total Return
            </CardTitle>
            <TrendingUp className="h-4 w-4 text-green-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-400">
              {analytics ? formatPercent(analytics.performance_metrics.total_return) : '0.00%'}
            </div>
            <p className="text-xs text-gray-400 mt-1">
              Annualized: {analytics ? formatPercent(analytics.performance_metrics.annualized_return) : '0.00%'}
            </p>
          </CardContent>
        </Card>

        {/* Sharpe Ratio */}
        <Card className="glow-hover">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">
              Sharpe Ratio
            </CardTitle>
            <BarChart3 className="h-4 w-4 text-ocean-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {analytics ? analytics.performance_metrics.sharpe_ratio.toFixed(2) : '0.00'}
            </div>
            <p className="text-xs text-gray-400 mt-1">
              Sortino: {analytics ? analytics.performance_metrics.sortino_ratio.toFixed(2) : '0.00'}
            </p>
          </CardContent>
        </Card>

        {/* Max Drawdown */}
        <Card className="glow-hover">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">
              Max Drawdown
            </CardTitle>
            <TrendingDown className="h-4 w-4 text-red-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-400">
              {analytics ? formatPercent(analytics.performance_metrics.max_drawdown) : '0.00%'}
            </div>
            <p className="text-xs text-gray-400 mt-1">
              Duration: {analytics ? analytics.performance_metrics.max_drawdown_duration : 0} days
            </p>
          </CardContent>
        </Card>

        {/* Win Rate */}
        <Card className="glow-hover">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">
              Win Rate
            </CardTitle>
            <Target className="h-4 w-4 text-wave-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-wave-400">
              {analytics ? formatPercent(analytics.performance_metrics.win_rate) : '0.00%'}
            </div>
            <p className="text-xs text-gray-400 mt-1">
              Profit Factor: {analytics ? analytics.performance_metrics.profit_factor.toFixed(2) : '0.00'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Chart */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Performance Overview</CardTitle>
            <CardDescription>Cumulative returns and daily performance</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64 flex items-center justify-center bg-white/5 rounded-lg">
              <div className="text-center">
                <BarChart3 className="w-12 h-12 text-gray-500 mx-auto mb-2" />
                <p className="text-gray-500 mb-1">Performance Chart</p>
                <p className="text-xs text-gray-600">Chart visualization would render here</p>
                <p className="text-xs text-gray-600 mt-2">
                  Showing {analytics?.chart_data.daily_returns.length} data points
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Market Regime Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="w-5 h-5 text-ocean-400" />
              <span>Market Regime Analysis</span>
            </CardTitle>
            <CardDescription>AI-powered market condition detection</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Current Regime */}
            <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
              <div className="flex items-center space-x-3">
                {analytics && getRegimeIcon(analytics.regime_data.current_regime)}
                <div>
                  <div className="font-medium text-white capitalize">
                    {analytics?.regime_data.current_regime.replace('_', ' ')} Market
                  </div>
                  <div className="text-sm text-gray-400">
                    Duration: {analytics?.regime_data.regime_duration} days
                  </div>
                </div>
              </div>
              <div className="text-right">
                <Badge className={analytics && getRegimeColor(analytics.regime_data.current_regime)}>
                  {analytics ? (analytics.regime_data.regime_confidence * 100).toFixed(0) : 0}% Confidence
                </Badge>
              </div>
            </div>

            {/* Regime History */}
            <div className="space-y-2">
              <h4 className="font-medium text-white">Recent Regime History</h4>
              {analytics?.regime_data.regime_history.map((regime, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                  <div className="flex items-center space-x-3">
                    {getRegimeIcon(regime.regime)}
                    <div>
                      <div className="text-sm font-medium text-white capitalize">
                        {regime.regime.replace('_', ' ')}
                      </div>
                      <div className="text-xs text-gray-400">
                        {regime.start_date} - {regime.end_date}
                      </div>
                    </div>
                  </div>
                  <div className={`text-sm font-medium ${
                    regime.performance >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {formatPercent(regime.performance)}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Risk Metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Shield className="w-5 h-5 text-green-400" />
              <span>Risk Metrics</span>
            </CardTitle>
            <CardDescription>Advanced risk analysis</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-400">Volatility</span>
                <span className="text-sm text-white">
                  {analytics ? formatPercent(analytics.performance_metrics.volatility) : '0.00%'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-400">Calmar Ratio</span>
                <span className="text-sm text-white">
                  {analytics ? analytics.performance_metrics.calmar_ratio.toFixed(2) : '0.00'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-400">Current Streak</span>
                <span className={`text-sm ${
                  analytics && analytics.performance_metrics.current_streak >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {analytics ? analytics.performance_metrics.current_streak : 0} wins
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-400">Max Win Streak</span>
                <span className="text-sm text-green-400">
                  {analytics ? analytics.performance_metrics.max_winning_streak : 0} wins
                </span>
              </div>
            </div>

            <div className="pt-4 border-t border-white/10">
              <div className="flex items-center space-x-2 mb-2">
                <CheckCircle className="w-4 h-4 text-green-400" />
                <span className="text-sm text-green-400">Risk Status: Low</span>
              </div>
              <p className="text-xs text-gray-400">
                Current risk metrics are within acceptable parameters
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Correlation Analysis */}
      <Card>
        <CardHeader>
          <CardTitle>Correlation Analysis</CardTitle>
          <CardDescription>Strategy and asset correlation matrix</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Strategy Correlations */}
            <div>
              <h4 className="font-medium text-white mb-3">Strategy Correlations</h4>
              <div className="space-y-2">
                {analytics?.correlation_data.strategy_correlations.map((corr, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                    <div className="text-sm">
                      <span className="text-white">{corr.strategy1}</span>
                      <span className="text-gray-400"> × </span>
                      <span className="text-white">{corr.strategy2}</span>
                    </div>
                    <Badge className={getCorrelationColor(corr.correlation)}>
                      {(corr.correlation * 100).toFixed(0)}%
                    </Badge>
                  </div>
                ))}
              </div>
            </div>

            {/* Asset Correlations */}
            <div>
              <h4 className="font-medium text-white mb-3">Asset Correlations</h4>
              <div className="space-y-2">
                {analytics?.correlation_data.asset_correlations.map((corr, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                    <div className="text-sm">
                      <span className="text-white">{corr.asset1}</span>
                      <span className="text-gray-400"> × </span>
                      <span className="text-white">{corr.asset2}</span>
                    </div>
                    <Badge className={getCorrelationColor(corr.correlation)}>
                      {(corr.correlation * 100).toFixed(0)}%
                    </Badge>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-6 p-4 bg-yellow-400/10 border border-yellow-400/20 rounded-lg">
            <div className="flex items-start space-x-2">
              <AlertTriangle className="w-5 h-5 text-yellow-400 mt-0.5 flex-shrink-0" />
              <div>
                <h5 className="font-medium text-yellow-400 mb-1">Correlation Warning</h5>
                <p className="text-sm text-yellow-400/80">
                  Some strategy pairs show high correlation (&gt;70%). Consider diversifying to reduce concentration risk.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Action Items */}
      <Card>
        <CardHeader>
          <CardTitle>Insights & Recommendations</CardTitle>
          <CardDescription>AI-powered actionable insights</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="glass p-4 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <TrendingUp className="w-5 h-5 text-green-400" />
                <span className="font-medium text-green-400">Performance</span>
              </div>
              <p className="text-sm text-gray-300">
                Strong Sharpe ratio indicates good risk-adjusted returns. Consider increasing allocation to top performers.
              </p>
            </div>
            
            <div className="glass p-4 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <Brain className="w-5 h-5 text-ocean-400" />
                <span className="font-medium text-ocean-400">Regime Adaptation</span>
              </div>
              <p className="text-sm text-gray-300">
                Current trending regime favors momentum strategies. Monitor for regime changes.
              </p>
            </div>
            
            <div className="glass p-4 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <Shield className="w-5 h-5 text-yellow-400" />
                <span className="font-medium text-yellow-400">Risk Management</span>
              </div>
              <p className="text-sm text-gray-300">
                High asset correlations detected. Consider adding uncorrelated assets for better diversification.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}