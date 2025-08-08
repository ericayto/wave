import React, { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Progress } from '../components/ui/progress'
import { Input } from '../components/ui/input'
import { Label } from '../components/ui/label'
import { formatCurrency, formatPercent } from '../lib/utils'
import { 
  PieChart,
  TrendingUp, 
  TrendingDown,
  BarChart3, 
  Shuffle,
  Target,
  Shield,
  AlertTriangle,
  CheckCircle,
  Settings,
  RefreshCw,
  Zap,
  Calculator,
  Scale,
  DollarSign,
  Percent,
  Clock
} from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select'
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '../components/ui/tabs'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../components/ui/dialog'

interface StrategyAllocation {
  strategy_id: string
  strategy_name: string
  current_allocation: number
  target_allocation: number
  performance: {
    total_return: number
    sharpe_ratio: number
    max_drawdown: number
    volatility: number
  }
  correlation_risk: number
  contribution_to_risk: number
}

interface PortfolioMetrics {
  total_value: number
  total_return: number
  portfolio_sharpe: number
  portfolio_volatility: number
  max_drawdown: number
  diversification_ratio: number
  tracking_error: number
  information_ratio: number
}

interface RiskMetrics {
  value_at_risk_95: number
  value_at_risk_99: number
  expected_shortfall_95: number
  expected_shortfall_99: number
  beta: number
  correlation_to_market: number
  maximum_correlation: number
  concentration_risk: number
}

interface RebalancingSignal {
  signal_id: string
  timestamp: string
  trigger: 'drift' | 'risk' | 'performance' | 'calendar'
  severity: 'low' | 'medium' | 'high'
  recommended_trades: Array<{
    strategy_id: string
    current_weight: number
    target_weight: number
    trade_amount: number
  }>
  expected_cost: number
  estimated_improvement: number
}

interface OptimizationResult {
  optimization_id: string
  method: 'max_sharpe' | 'min_variance' | 'risk_parity' | 'black_litterman'
  optimized_weights: Record<string, number>
  expected_return: number
  expected_volatility: number
  expected_sharpe: number
  improvement_vs_current: {
    return_improvement: number
    risk_reduction: number
    sharpe_improvement: number
  }
}

// Mock data
const mockStrategies: StrategyAllocation[] = [
  {
    strategy_id: 'momentum_1',
    strategy_name: 'Momentum Strategy',
    current_allocation: 0.35,
    target_allocation: 0.30,
    performance: {
      total_return: 0.124,
      sharpe_ratio: 1.34,
      max_drawdown: -0.087,
      volatility: 0.145
    },
    correlation_risk: 0.67,
    contribution_to_risk: 0.42
  },
  {
    strategy_id: 'mean_reversion_1',
    strategy_name: 'Mean Reversion',
    current_allocation: 0.25,
    target_allocation: 0.25,
    performance: {
      total_return: 0.089,
      sharpe_ratio: 0.98,
      max_drawdown: -0.054,
      volatility: 0.112
    },
    correlation_risk: -0.23,
    contribution_to_risk: 0.18
  },
  {
    strategy_id: 'arbitrage_1',
    strategy_name: 'Arbitrage Strategy',
    current_allocation: 0.20,
    target_allocation: 0.25,
    performance: {
      total_return: 0.067,
      sharpe_ratio: 1.87,
      max_drawdown: -0.023,
      volatility: 0.087
    },
    correlation_risk: 0.12,
    contribution_to_risk: 0.15
  },
  {
    strategy_id: 'breakout_1',
    strategy_name: 'Breakout Strategy',
    current_allocation: 0.20,
    target_allocation: 0.20,
    performance: {
      total_return: 0.156,
      sharpe_ratio: 1.23,
      max_drawdown: -0.112,
      volatility: 0.187
    },
    correlation_risk: 0.78,
    contribution_to_risk: 0.25
  }
]

const fetchPortfolioMetrics = async (): Promise<PortfolioMetrics> => {
  await new Promise(resolve => setTimeout(resolve, 500))
  return {
    total_value: 105420.50,
    total_return: 0.0542,
    portfolio_sharpe: 1.28,
    portfolio_volatility: 0.134,
    max_drawdown: -0.067,
    diversification_ratio: 1.34,
    tracking_error: 0.045,
    information_ratio: 0.67
  }
}

const fetchRiskMetrics = async (): Promise<RiskMetrics> => {
  await new Promise(resolve => setTimeout(resolve, 400))
  return {
    value_at_risk_95: -0.0234,
    value_at_risk_99: -0.0387,
    expected_shortfall_95: -0.0298,
    expected_shortfall_99: -0.0445,
    beta: 0.87,
    correlation_to_market: 0.67,
    maximum_correlation: 0.78,
    concentration_risk: 0.35
  }
}

const fetchRebalancingSignals = async (): Promise<RebalancingSignal[]> => {
  await new Promise(resolve => setTimeout(resolve, 300))
  return [
    {
      signal_id: 'rebal_001',
      timestamp: new Date().toISOString(),
      trigger: 'drift',
      severity: 'medium',
      recommended_trades: [
        {
          strategy_id: 'momentum_1',
          current_weight: 0.35,
          target_weight: 0.30,
          trade_amount: -5271.03
        },
        {
          strategy_id: 'arbitrage_1',
          current_weight: 0.20,
          target_weight: 0.25,
          trade_amount: 5271.03
        }
      ],
      expected_cost: 15.23,
      estimated_improvement: 0.0034
    }
  ]
}

export const PortfolioManagement: React.FC = () => {
  const [optimizationMethod, setOptimizationMethod] = useState<string>('max_sharpe')
  const [isOptimizationDialogOpen, setIsOptimizationDialogOpen] = useState(false)
  const [rebalancingMode, setRebalancingMode] = useState<'manual' | 'automatic'>('manual')
  const [selectedTimeframe, setSelectedTimeframe] = useState('30d')

  const { data: portfolioMetrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['portfolio-metrics'],
    queryFn: fetchPortfolioMetrics,
    refetchInterval: 30000,
  })

  const { data: riskMetrics, isLoading: riskLoading } = useQuery({
    queryKey: ['risk-metrics'],
    queryFn: fetchRiskMetrics,
    refetchInterval: 60000,
  })

  const { data: rebalancingSignals, isLoading: signalsLoading } = useQuery({
    queryKey: ['rebalancing-signals'],
    queryFn: fetchRebalancingSignals,
    refetchInterval: 300000, // 5 minutes
  })

  const optimizePortfolioMutation = useMutation({
    mutationFn: async (config: { method: string, constraints?: any }) => {
      await new Promise(resolve => setTimeout(resolve, 2000))
      return {
        optimization_id: 'opt_' + Date.now(),
        method: config.method,
        optimized_weights: {
          momentum_1: 0.28,
          mean_reversion_1: 0.27,
          arbitrage_1: 0.28,
          breakout_1: 0.17
        },
        expected_return: 0.089,
        expected_volatility: 0.121,
        expected_sharpe: 1.47,
        improvement_vs_current: {
          return_improvement: 0.0087,
          risk_reduction: 0.0132,
          sharpe_improvement: 0.19
        }
      } as OptimizationResult
    },
    onSuccess: () => {
      setIsOptimizationDialogOpen(false)
    },
  })

  const executeRebalancingMutation = useMutation({
    mutationFn: async (signalId: string) => {
      await new Promise(resolve => setTimeout(resolve, 1500))
      return { success: true, executed_trades: 2 }
    },
  })

  const getAllocationColor = (current: number, target: number) => {
    const diff = Math.abs(current - target)
    if (diff > 0.05) return 'text-red-400'
    if (diff > 0.02) return 'text-yellow-400'
    return 'text-green-400'
  }

  const getRiskColor = (value: number, thresholds: [number, number]) => {
    if (Math.abs(value) > thresholds[1]) return 'text-red-400'
    if (Math.abs(value) > thresholds[0]) return 'text-yellow-400'
    return 'text-green-400'
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'text-red-400 bg-red-400/10'
      case 'medium': return 'text-yellow-400 bg-yellow-400/10'
      case 'low': return 'text-green-400 bg-green-400/10'
      default: return 'text-gray-400 bg-gray-400/10'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Portfolio Management</h1>
          <p className="text-gray-400 mt-2">Multi-strategy allocation, optimization, and risk management</p>
        </div>
        <div className="flex items-center space-x-4">
          <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="7d">7 Days</SelectItem>
              <SelectItem value="30d">30 Days</SelectItem>
              <SelectItem value="90d">90 Days</SelectItem>
            </SelectContent>
          </Select>
          <Dialog open={isOptimizationDialogOpen} onOpenChange={setIsOptimizationDialogOpen}>
            <DialogTrigger asChild>
              <Button className="bg-gradient-to-r from-ocean-500 to-wave-500">
                <Calculator className="w-4 h-4 mr-2" />
                Optimize Portfolio
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Portfolio Optimization</DialogTitle>
                <DialogDescription>
                  Select optimization method and constraints
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <Label>Optimization Method</Label>
                  <Select value={optimizationMethod} onValueChange={setOptimizationMethod}>
                    <SelectTrigger className="mt-1">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="max_sharpe">Maximum Sharpe Ratio</SelectItem>
                      <SelectItem value="min_variance">Minimum Variance</SelectItem>
                      <SelectItem value="risk_parity">Risk Parity</SelectItem>
                      <SelectItem value="black_litterman">Black-Litterman</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <DialogFooter>
                <Button 
                  onClick={() => optimizePortfolioMutation.mutate({ method: optimizationMethod })}
                  disabled={optimizePortfolioMutation.isPending}
                >
                  {optimizePortfolioMutation.isPending ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      Optimizing...
                    </>
                  ) : (
                    'Run Optimization'
                  )}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="allocation">Allocation</TabsTrigger>
          <TabsTrigger value="optimization">Optimization</TabsTrigger>
          <TabsTrigger value="rebalancing">Rebalancing</TabsTrigger>
          <TabsTrigger value="risk">Risk Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Portfolio Overview Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="glow-hover">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">
                  Portfolio Value
                </CardTitle>
                <DollarSign className="h-4 w-4 text-green-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-white">
                  {portfolioMetrics ? formatCurrency(portfolioMetrics.total_value) : '$0.00'}
                </div>
                <p className="text-xs text-green-400 mt-1">
                  {portfolioMetrics ? formatPercent(portfolioMetrics.total_return, true) : '+0.00%'}
                </p>
              </CardContent>
            </Card>

            <Card className="glow-hover">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">
                  Portfolio Sharpe
                </CardTitle>
                <BarChart3 className="h-4 w-4 text-ocean-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-white">
                  {portfolioMetrics ? portfolioMetrics.portfolio_sharpe.toFixed(2) : '0.00'}
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  Volatility: {portfolioMetrics ? formatPercent(portfolioMetrics.portfolio_volatility) : '0.00%'}
                </p>
              </CardContent>
            </Card>

            <Card className="glow-hover">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">
                  Max Drawdown
                </CardTitle>
                <TrendingDown className="h-4 w-4 text-red-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-red-400">
                  {portfolioMetrics ? formatPercent(portfolioMetrics.max_drawdown) : '0.00%'}
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  Diversification: {portfolioMetrics ? portfolioMetrics.diversification_ratio.toFixed(2) : '0.00'}
                </p>
              </CardContent>
            </Card>

            <Card className="glow-hover">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">
                  Information Ratio
                </CardTitle>
                <Target className="h-4 w-4 text-wave-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-wave-400">
                  {portfolioMetrics ? portfolioMetrics.information_ratio.toFixed(2) : '0.00'}
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  Tracking Error: {portfolioMetrics ? formatPercent(portfolioMetrics.tracking_error) : '0.00%'}
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Portfolio Composition */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Portfolio Allocation</CardTitle>
                <CardDescription>Current vs target strategy weights</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {mockStrategies.map((strategy) => (
                    <div key={strategy.strategy_id} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-white">{strategy.strategy_name}</span>
                        <div className="flex items-center space-x-2">
                          <span className={`text-sm ${getAllocationColor(strategy.current_allocation, strategy.target_allocation)}`}>
                            {formatPercent(strategy.current_allocation)}
                          </span>
                          <span className="text-xs text-gray-400">
                            (Target: {formatPercent(strategy.target_allocation)})
                          </span>
                        </div>
                      </div>
                      <div className="flex space-x-2">
                        <div className="flex-1">
                          <Progress value={strategy.current_allocation * 100} className="h-2" />
                        </div>
                        <div className="w-16 text-xs text-gray-400 text-right">
                          {strategy.current_allocation > strategy.target_allocation ? 'Over' : 'Under'}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Strategy Performance</CardTitle>
                <CardDescription>Individual strategy metrics</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {mockStrategies.map((strategy) => (
                    <div key={strategy.strategy_id} className="p-3 bg-white/5 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-white">{strategy.strategy_name}</span>
                        <Badge className="text-green-400 bg-green-400/10">
                          {formatPercent(strategy.performance.total_return, true)}
                        </Badge>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        <div>
                          <div className="text-gray-400">Sharpe</div>
                          <div className="text-white">{strategy.performance.sharpe_ratio.toFixed(2)}</div>
                        </div>
                        <div>
                          <div className="text-gray-400">Max DD</div>
                          <div className="text-red-400">{formatPercent(strategy.performance.max_drawdown)}</div>
                        </div>
                        <div>
                          <div className="text-gray-400">Volatility</div>
                          <div className="text-white">{formatPercent(strategy.performance.volatility)}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Rebalancing Alerts */}
          {!signalsLoading && rebalancingSignals && rebalancingSignals.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Shuffle className="w-5 h-5 text-yellow-400" />
                  <span>Rebalancing Alerts</span>
                </CardTitle>
                <CardDescription>Portfolio drift and rebalancing recommendations</CardDescription>
              </CardHeader>
              <CardContent>
                {rebalancingSignals.map((signal) => (
                  <div key={signal.signal_id} className="p-4 bg-yellow-400/10 border border-yellow-400/20 rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <Badge className={getSeverityColor(signal.severity)}>
                          {signal.severity.toUpperCase()}
                        </Badge>
                        <span className="text-sm text-white capitalize">
                          {signal.trigger} trigger
                        </span>
                      </div>
                      <Button 
                        size="sm"
                        onClick={() => executeRebalancingMutation.mutate(signal.signal_id)}
                        disabled={executeRebalancingMutation.isPending}
                      >
                        {executeRebalancingMutation.isPending ? (
                          <RefreshCw className="w-3 h-3 animate-spin" />
                        ) : (
                          'Execute'
                        )}
                      </Button>
                    </div>
                    
                    <div className="space-y-2 text-sm">
                      {signal.recommended_trades.map((trade, index) => (
                        <div key={index} className="flex items-center justify-between p-2 bg-white/5 rounded">
                          <span className="text-gray-300">
                            {mockStrategies.find(s => s.strategy_id === trade.strategy_id)?.strategy_name}
                          </span>
                          <div className="flex items-center space-x-2">
                            <span className="text-gray-400">
                              {formatPercent(trade.current_weight)} → {formatPercent(trade.target_weight)}
                            </span>
                            <span className={`font-medium ${trade.trade_amount > 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {formatCurrency(trade.trade_amount, true)}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                    
                    <div className="mt-3 pt-3 border-t border-white/10 flex items-center justify-between text-xs">
                      <span className="text-gray-400">
                        Expected Cost: {formatCurrency(signal.expected_cost)}
                      </span>
                      <span className="text-green-400">
                        Est. Improvement: {formatPercent(signal.estimated_improvement, true)}
                      </span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="allocation" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Allocation Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Allocation Visualization</CardTitle>
                <CardDescription>Portfolio composition pie chart</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center bg-white/5 rounded-lg">
                  <div className="text-center">
                    <PieChart className="w-12 h-12 text-gray-500 mx-auto mb-2" />
                    <p className="text-gray-500 mb-1">Portfolio Allocation Chart</p>
                    <p className="text-xs text-gray-600">Interactive pie chart would render here</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Manual Allocation Adjustment */}
            <Card>
              <CardHeader>
                <CardTitle>Manual Allocation</CardTitle>
                <CardDescription>Adjust target allocations</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {mockStrategies.map((strategy) => (
                  <div key={strategy.strategy_id} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">{strategy.strategy_name}</Label>
                      <span className="text-xs text-gray-400">
                        Current: {formatPercent(strategy.current_allocation)}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Input 
                        type="number" 
                        step="0.01"
                        min="0"
                        max="1"
                        defaultValue={strategy.target_allocation}
                        className="flex-1"
                      />
                      <span className="text-xs text-gray-400 w-8">%</span>
                    </div>
                  </div>
                ))}
                
                <div className="pt-4 border-t border-white/10 flex justify-between">
                  <span className="text-sm text-gray-400">Total:</span>
                  <span className="text-sm text-white">100.0%</span>
                </div>
                
                <Button className="w-full" size="sm">
                  <Settings className="w-4 h-4 mr-2" />
                  Apply Changes
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Correlation Matrix */}
          <Card>
            <CardHeader>
              <CardTitle>Strategy Correlation Matrix</CardTitle>
              <CardDescription>Cross-correlation between strategies</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-5 gap-2 text-xs">
                <div></div>
                {mockStrategies.map((strategy) => (
                  <div key={strategy.strategy_id} className="text-center text-gray-400 p-2">
                    {strategy.strategy_name.split(' ')[0]}
                  </div>
                ))}
                
                {mockStrategies.map((strategy1) => (
                  <React.Fragment key={strategy1.strategy_id}>
                    <div className="text-gray-400 p-2">{strategy1.strategy_name.split(' ')[0]}</div>
                    {mockStrategies.map((strategy2) => {
                      const correlation = strategy1.strategy_id === strategy2.strategy_id ? 1.0 : 
                                        Math.random() * 0.8 - 0.4 // Mock correlation
                      const color = Math.abs(correlation) > 0.7 ? 'bg-red-400/20 text-red-400' :
                                   Math.abs(correlation) > 0.3 ? 'bg-yellow-400/20 text-yellow-400' :
                                   'bg-green-400/20 text-green-400'
                      
                      return (
                        <div key={strategy2.strategy_id} className={`p-2 text-center rounded ${color}`}>
                          {correlation.toFixed(2)}
                        </div>
                      )
                    })}
                  </React.Fragment>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="optimization" className="space-y-6">
          {optimizePortfolioMutation.data && (
            <Card>
              <CardHeader>
                <CardTitle>Optimization Results</CardTitle>
                <CardDescription>Latest portfolio optimization results</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="p-4 bg-green-400/10 border border-green-400/20 rounded-lg">
                    <div className="text-sm text-green-400 mb-1">Expected Return</div>
                    <div className="text-xl font-bold text-white">
                      {formatPercent(optimizePortfolioMutation.data.expected_return)}
                    </div>
                    <div className="text-xs text-green-400">
                      +{formatPercent(optimizePortfolioMutation.data.improvement_vs_current.return_improvement)} improvement
                    </div>
                  </div>
                  
                  <div className="p-4 bg-blue-400/10 border border-blue-400/20 rounded-lg">
                    <div className="text-sm text-blue-400 mb-1">Expected Volatility</div>
                    <div className="text-xl font-bold text-white">
                      {formatPercent(optimizePortfolioMutation.data.expected_volatility)}
                    </div>
                    <div className="text-xs text-blue-400">
                      -{formatPercent(optimizePortfolioMutation.data.improvement_vs_current.risk_reduction)} reduction
                    </div>
                  </div>
                  
                  <div className="p-4 bg-ocean-400/10 border border-ocean-400/20 rounded-lg">
                    <div className="text-sm text-ocean-400 mb-1">Expected Sharpe</div>
                    <div className="text-xl font-bold text-white">
                      {optimizePortfolioMutation.data.expected_sharpe.toFixed(2)}
                    </div>
                    <div className="text-xs text-ocean-400">
                      +{optimizePortfolioMutation.data.improvement_vs_current.sharpe_improvement.toFixed(2)} improvement
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium text-white mb-3">Optimized Weights</h4>
                  <div className="space-y-2">
                    {Object.entries(optimizePortfolioMutation.data.optimized_weights).map(([strategyId, weight]) => {
                      const strategy = mockStrategies.find(s => s.strategy_id === strategyId)
                      const currentWeight = strategy?.current_allocation || 0
                      const change = weight - currentWeight
                      
                      return (
                        <div key={strategyId} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                          <span className="text-white">{strategy?.strategy_name || strategyId}</span>
                          <div className="flex items-center space-x-3">
                            <span className="text-gray-400">{formatPercent(currentWeight)} →</span>
                            <span className="text-white font-medium">{formatPercent(weight)}</span>
                            <span className={`text-sm ${change > 0 ? 'text-green-400' : 'text-red-400'}`}>
                              ({formatPercent(change, true)})
                            </span>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
                
                <div className="flex space-x-4">
                  <Button className="flex-1">
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Apply Optimization
                  </Button>
                  <Button variant="outline">
                    View Details
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Optimization Methods */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Modern Portfolio Theory</CardTitle>
                <CardDescription>Classic optimization approaches</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <Button variant="outline" className="w-full justify-start" size="sm">
                    <Target className="w-4 h-4 mr-2" />
                    Maximum Sharpe Ratio
                  </Button>
                  <Button variant="outline" className="w-full justify-start" size="sm">
                    <Shield className="w-4 h-4 mr-2" />
                    Minimum Variance
                  </Button>
                  <Button variant="outline" className="w-full justify-start" size="sm">
                    <Scale className="w-4 h-4 mr-2" />
                    Risk Parity
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Advanced Methods</CardTitle>
                <CardDescription>Enhanced optimization techniques</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <Button variant="outline" className="w-full justify-start" size="sm">
                    <Zap className="w-4 h-4 mr-2" />
                    Black-Litterman
                  </Button>
                  <Button variant="outline" className="w-full justify-start" size="sm">
                    <BarChart3 className="w-4 h-4 mr-2" />
                    Factor-Based
                  </Button>
                  <Button variant="outline" className="w-full justify-start" size="sm">
                    <Calculator className="w-4 h-4 mr-2" />
                    Multi-Objective
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="rebalancing" className="space-y-6">
          {/* Rebalancing Settings */}
          <Card>
            <CardHeader>
              <CardTitle>Rebalancing Configuration</CardTitle>
              <CardDescription>Automated rebalancing rules and triggers</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label>Rebalancing Mode</Label>
                    <Select value={rebalancingMode} onValueChange={(value: any) => setRebalancingMode(value)}>
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="manual">Manual</SelectItem>
                        <SelectItem value="automatic">Automatic</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div>
                    <Label>Drift Threshold</Label>
                    <Input type="number" step="0.01" defaultValue="0.05" className="mt-1" />
                    <p className="text-xs text-gray-400 mt-1">Trigger rebalancing when allocation drifts by this amount</p>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <Label>Rebalancing Frequency</Label>
                    <Select defaultValue="weekly">
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="daily">Daily</SelectItem>
                        <SelectItem value="weekly">Weekly</SelectItem>
                        <SelectItem value="monthly">Monthly</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div>
                    <Label>Maximum Trade Size</Label>
                    <Input type="number" step="1000" defaultValue="10000" className="mt-1" />
                    <p className="text-xs text-gray-400 mt-1">Maximum dollar amount per rebalancing trade</p>
                  </div>
                </div>
              </div>
              
              <div className="pt-4 border-t border-white/10">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium text-white">Automatic Rebalancing</h4>
                    <p className="text-sm text-gray-400">Execute rebalancing trades automatically</p>
                  </div>
                  <Button 
                    variant={rebalancingMode === 'automatic' ? 'default' : 'outline'}
                    size="sm"
                  >
                    {rebalancingMode === 'automatic' ? 'Enabled' : 'Enable'}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Rebalancing History */}
          <Card>
            <CardHeader>
              <CardTitle>Rebalancing History</CardTitle>
              <CardDescription>Recent rebalancing activities</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <div>
                        <div className="text-sm font-medium text-white">
                          Drift Rebalancing #{i}
                        </div>
                        <div className="text-xs text-gray-400">
                          {new Date(Date.now() - i * 7 * 24 * 60 * 60 * 1000).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-white">2 trades</div>
                      <div className="text-xs text-gray-400">Cost: $12.45</div>
                    </div>
                  </div>
                ))}
                
                <div className="text-center py-4">
                  <Button variant="outline" size="sm">
                    <Clock className="w-4 h-4 mr-2" />
                    View All History
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="risk" className="space-y-6">
          {/* Risk Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="glow-hover">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">
                  VaR (95%)
                </CardTitle>
                <AlertTriangle className="h-4 w-4 text-yellow-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-yellow-400">
                  {riskMetrics ? formatPercent(riskMetrics.value_at_risk_95) : '0.00%'}
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  99%: {riskMetrics ? formatPercent(riskMetrics.value_at_risk_99) : '0.00%'}
                </p>
              </CardContent>
            </Card>

            <Card className="glow-hover">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">
                  Expected Shortfall
                </CardTitle>
                <TrendingDown className="h-4 w-4 text-red-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-red-400">
                  {riskMetrics ? formatPercent(riskMetrics.expected_shortfall_95) : '0.00%'}
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  99%: {riskMetrics ? formatPercent(riskMetrics.expected_shortfall_99) : '0.00%'}
                </p>
              </CardContent>
            </Card>

            <Card className="glow-hover">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">
                  Beta
                </CardTitle>
                <BarChart3 className="h-4 w-4 text-ocean-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-white">
                  {riskMetrics ? riskMetrics.beta.toFixed(2) : '0.00'}
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  Market Correlation: {riskMetrics ? (riskMetrics.correlation_to_market * 100).toFixed(0) : 0}%
                </p>
              </CardContent>
            </Card>

            <Card className="glow-hover">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">
                  Concentration Risk
                </CardTitle>
                <Shield className="h-4 w-4 text-wave-400" />
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-bold ${
                  riskMetrics && riskMetrics.concentration_risk > 0.4 ? 'text-red-400' : 
                  riskMetrics && riskMetrics.concentration_risk > 0.25 ? 'text-yellow-400' : 'text-green-400'
                }`}>
                  {riskMetrics ? formatPercent(riskMetrics.concentration_risk) : '0.00%'}
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  Max Correlation: {riskMetrics ? (riskMetrics.maximum_correlation * 100).toFixed(0) : 0}%
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Risk Contribution */}
          <Card>
            <CardHeader>
              <CardTitle>Risk Contribution by Strategy</CardTitle>
              <CardDescription>How each strategy contributes to portfolio risk</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockStrategies.map((strategy) => (
                  <div key={strategy.strategy_id} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-white">{strategy.strategy_name}</span>
                      <div className="flex items-center space-x-3">
                        <span className="text-sm text-gray-400">
                          {formatPercent(strategy.contribution_to_risk)}
                        </span>
                        <span className={`text-xs ${getRiskColor(strategy.correlation_risk, [0.3, 0.7])}`}>
                          Corr: {(strategy.correlation_risk * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    <Progress value={strategy.contribution_to_risk * 100} className="h-2" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Risk Alerts */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <AlertTriangle className="w-5 h-5 text-yellow-400" />
                <span>Risk Alerts</span>
              </CardTitle>
              <CardDescription>Current risk warnings and recommendations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="p-3 bg-yellow-400/10 border border-yellow-400/20 rounded-lg">
                  <div className="flex items-start space-x-2">
                    <AlertTriangle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <h5 className="font-medium text-yellow-400 mb-1">High Correlation Warning</h5>
                      <p className="text-sm text-yellow-400/80">
                        Momentum and Breakout strategies show high correlation (78%). Consider reducing allocation to one of these strategies.
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="p-3 bg-green-400/10 border border-green-400/20 rounded-lg">
                  <div className="flex items-start space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <h5 className="font-medium text-green-400 mb-1">Diversification Good</h5>
                      <p className="text-sm text-green-400/80">
                        Portfolio diversification ratio (1.34) indicates good risk distribution across strategies.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}