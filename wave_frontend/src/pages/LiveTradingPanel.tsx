import React, { useState, useEffect } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Progress } from '../components/ui/progress'
import { Input } from '../components/ui/input'
import { Label } from '../components/ui/label'
import { formatCurrency, formatPercent } from '../lib/utils'
import { 
  Play,
  Square, 
  AlertTriangle,
  Shield,
  Zap,
  Activity,
  DollarSign,
  TrendingUp,
  TrendingDown,
  Clock,
  Settings,
  RefreshCw,
  CheckCircle,
  XCircle,
  Pause,
  SkipForward,
  Eye,
  Bell,
  Lock,
  Unlock,
  AlertCircle,
  Target,
  BarChart3
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
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from '../components/ui/alert'

interface LiveTradingStatus {
  is_active: boolean
  mode: 'paper' | 'live'
  started_at?: string
  uptime_seconds: number
  total_trades_today: number
  active_strategies: number
  current_capital: number
  daily_pnl: number
  daily_pnl_pct: number
}

interface LiveOrder {
  order_id: string
  strategy_id: string
  symbol: string
  side: 'buy' | 'sell'
  order_type: 'market' | 'limit' | 'stop_loss' | 'take_profit'
  quantity: number
  price?: number
  filled_quantity: number
  remaining_quantity: number
  status: 'pending' | 'partially_filled' | 'filled' | 'cancelled' | 'rejected'
  timestamp: string
  estimated_cost: number
}

interface PositionSnapshot {
  symbol: string
  strategy_id: string
  quantity: number
  average_price: number
  market_price: number
  market_value: number
  unrealized_pnl: number
  unrealized_pnl_pct: number
  duration_hours: number
  risk_score: number
}

interface EmergencyStop {
  stop_id: string
  trigger_type: 'drawdown' | 'loss_limit' | 'manual' | 'compliance' | 'technical'
  trigger_value: number
  threshold: number
  activated_at: string
  is_active: boolean
  affected_strategies: string[]
  recovery_conditions: string[]
}

interface ComplianceCheck {
  check_id: string
  rule_name: string
  severity: 'info' | 'warning' | 'critical'
  status: 'passing' | 'failing'
  current_value: number
  limit_value: number
  description: string
  last_checked: string
}

interface RiskLimit {
  limit_id: string
  name: string
  limit_type: 'position_size' | 'daily_loss' | 'concentration' | 'leverage' | 'drawdown'
  current_value: number
  limit_value: number
  utilization_pct: number
  status: 'safe' | 'warning' | 'breach'
}

const mockLiveTradingStatus: LiveTradingStatus = {
  is_active: false,
  mode: 'paper',
  uptime_seconds: 0,
  total_trades_today: 0,
  active_strategies: 3,
  current_capital: 100000,
  daily_pnl: 0,
  daily_pnl_pct: 0
}

const mockLiveOrders: LiveOrder[] = [
  {
    order_id: 'ord_001',
    strategy_id: 'momentum_1',
    symbol: 'BTC-USD',
    side: 'buy',
    order_type: 'limit',
    quantity: 0.5,
    price: 43250,
    filled_quantity: 0.3,
    remaining_quantity: 0.2,
    status: 'partially_filled',
    timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
    estimated_cost: 21625
  },
  {
    order_id: 'ord_002',
    strategy_id: 'mean_reversion_1',
    symbol: 'ETH-USD',
    side: 'sell',
    order_type: 'stop_loss',
    quantity: 2.0,
    price: 2480,
    filled_quantity: 0,
    remaining_quantity: 2.0,
    status: 'pending',
    timestamp: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
    estimated_cost: 4960
  }
]

const mockPositions: PositionSnapshot[] = [
  {
    symbol: 'BTC-USD',
    strategy_id: 'momentum_1',
    quantity: 0.8,
    average_price: 42800,
    market_price: 43100,
    market_value: 34480,
    unrealized_pnl: 240,
    unrealized_pnl_pct: 0.007,
    duration_hours: 4.2,
    risk_score: 0.65
  },
  {
    symbol: 'ETH-USD',
    strategy_id: 'arbitrage_1',
    quantity: 5.0,
    average_price: 2520,
    market_price: 2495,
    market_value: 12475,
    unrealized_pnl: -125,
    unrealized_pnl_pct: -0.01,
    duration_hours: 1.8,
    risk_score: 0.45
  }
]

const mockRiskLimits: RiskLimit[] = [
  {
    limit_id: 'limit_001',
    name: 'Daily Loss Limit',
    limit_type: 'daily_loss',
    current_value: 850,
    limit_value: 2000,
    utilization_pct: 42.5,
    status: 'safe'
  },
  {
    limit_id: 'limit_002',
    name: 'Position Concentration',
    limit_type: 'concentration',
    current_value: 35.2,
    limit_value: 40,
    utilization_pct: 88.0,
    status: 'warning'
  },
  {
    limit_id: 'limit_003',
    name: 'Maximum Drawdown',
    limit_type: 'drawdown',
    current_value: 3.4,
    limit_value: 5.0,
    utilization_pct: 68.0,
    status: 'safe'
  }
]

const mockComplianceChecks: ComplianceCheck[] = [
  {
    check_id: 'comp_001',
    rule_name: 'Pattern Day Trading',
    severity: 'warning',
    status: 'passing',
    current_value: 2,
    limit_value: 3,
    description: 'Day trades remaining this week',
    last_checked: new Date().toISOString()
  },
  {
    check_id: 'comp_002',
    rule_name: 'Position Size Limits',
    severity: 'critical',
    status: 'passing',
    current_value: 35000,
    limit_value: 50000,
    description: 'Maximum single position value',
    last_checked: new Date().toISOString()
  }
]

const fetchLiveTradingStatus = async (): Promise<LiveTradingStatus> => {
  await new Promise(resolve => setTimeout(resolve, 300))
  return {
    ...mockLiveTradingStatus,
    uptime_seconds: mockLiveTradingStatus.is_active ? Math.floor(Math.random() * 3600) : 0,
    total_trades_today: Math.floor(Math.random() * 15),
    daily_pnl: Math.random() * 400 - 200,
    daily_pnl_pct: (Math.random() * 0.004) - 0.002
  }
}

const fetchLiveOrders = async (): Promise<LiveOrder[]> => {
  await new Promise(resolve => setTimeout(resolve, 200))
  return mockLiveOrders
}

const fetchPositions = async (): Promise<PositionSnapshot[]> => {
  await new Promise(resolve => setTimeout(resolve, 250))
  return mockPositions
}

const fetchRiskLimits = async (): Promise<RiskLimit[]> => {
  await new Promise(resolve => setTimeout(resolve, 150))
  return mockRiskLimits
}

const fetchComplianceChecks = async (): Promise<ComplianceCheck[]> => {
  await new Promise(resolve => setTimeout(resolve, 100))
  return mockComplianceChecks
}

export const LiveTradingPanel: React.FC = () => {
  const [emergencyStopDialogOpen, setEmergencyStopDialogOpen] = useState(false)
  const [emergencyStopReason, setEmergencyStopReason] = useState('')
  const [isLiveModeEnabled, setIsLiveModeEnabled] = useState(false)
  const [confirmationInput, setConfirmationInput] = useState('')

  const { data: tradingStatus, refetch: refetchStatus } = useQuery({
    queryKey: ['live-trading-status'],
    queryFn: fetchLiveTradingStatus,
    refetchInterval: 2000, // Update every 2 seconds
  })

  const { data: liveOrders } = useQuery({
    queryKey: ['live-orders'],
    queryFn: fetchLiveOrders,
    refetchInterval: 3000, // Update every 3 seconds
  })

  const { data: positions } = useQuery({
    queryKey: ['positions'],
    queryFn: fetchPositions,
    refetchInterval: 5000, // Update every 5 seconds
  })

  const { data: riskLimits } = useQuery({
    queryKey: ['risk-limits'],
    queryFn: fetchRiskLimits,
    refetchInterval: 10000, // Update every 10 seconds
  })

  const { data: complianceChecks } = useQuery({
    queryKey: ['compliance-checks'],
    queryFn: fetchComplianceChecks,
    refetchInterval: 30000, // Update every 30 seconds
  })

  const startTradingMutation = useMutation({
    mutationFn: async (mode: 'paper' | 'live') => {
      await new Promise(resolve => setTimeout(resolve, 2000))
      return { success: true, mode }
    },
    onSuccess: () => {
      refetchStatus()
    },
  })

  const stopTradingMutation = useMutation({
    mutationFn: async () => {
      await new Promise(resolve => setTimeout(resolve, 1000))
      return { success: true }
    },
    onSuccess: () => {
      refetchStatus()
    },
  })

  const emergencyStopMutation = useMutation({
    mutationFn: async (reason: string) => {
      await new Promise(resolve => setTimeout(resolve, 1500))
      return { success: true, reason }
    },
    onSuccess: () => {
      setEmergencyStopDialogOpen(false)
      refetchStatus()
    },
  })

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60
    return `${hours}h ${minutes}m ${secs}s`
  }

  const getOrderStatusColor = (status: string) => {
    switch (status) {
      case 'filled': return 'text-green-400 bg-green-400/10'
      case 'partially_filled': return 'text-blue-400 bg-blue-400/10'
      case 'pending': return 'text-yellow-400 bg-yellow-400/10'
      case 'cancelled': return 'text-gray-400 bg-gray-400/10'
      case 'rejected': return 'text-red-400 bg-red-400/10'
      default: return 'text-gray-400 bg-gray-400/10'
    }
  }

  const getRiskStatusColor = (status: string) => {
    switch (status) {
      case 'safe': return 'text-green-400 bg-green-400/10'
      case 'warning': return 'text-yellow-400 bg-yellow-400/10'
      case 'breach': return 'text-red-400 bg-red-400/10'
      default: return 'text-gray-400 bg-gray-400/10'
    }
  }

  const getComplianceStatusColor = (severity: string, status: string) => {
    if (status === 'failing') return 'text-red-400 bg-red-400/10'
    switch (severity) {
      case 'critical': return 'text-orange-400 bg-orange-400/10'
      case 'warning': return 'text-yellow-400 bg-yellow-400/10'
      default: return 'text-green-400 bg-green-400/10'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header with Emergency Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Live Trading Panel</h1>
          <p className="text-gray-400 mt-2">Real-time trading controls and monitoring</p>
        </div>
        <div className="flex items-center space-x-4">
          {/* Mode Toggle */}
          <div className="flex items-center space-x-2">
            <Label className="text-sm text-gray-400">Mode:</Label>
            <Button 
              variant={tradingStatus?.mode === 'live' ? 'destructive' : 'outline'}
              size="sm"
              onClick={() => setIsLiveModeEnabled(!isLiveModeEnabled)}
              disabled={tradingStatus?.is_active}
            >
              {isLiveModeEnabled ? <Lock className="w-4 h-4 mr-1" /> : <Unlock className="w-4 h-4 mr-1" />}
              {tradingStatus?.mode === 'live' ? 'LIVE' : 'PAPER'}
            </Button>
          </div>

          {/* Emergency Stop */}
          <Dialog open={emergencyStopDialogOpen} onOpenChange={setEmergencyStopDialogOpen}>
            <DialogTrigger asChild>
              <Button 
                variant="destructive" 
                size="sm"
                disabled={!tradingStatus?.is_active}
                className="bg-red-600 hover:bg-red-700"
              >
                <AlertTriangle className="w-4 h-4 mr-2" />
                EMERGENCY STOP
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle className="flex items-center space-x-2 text-red-400">
                  <AlertTriangle className="w-5 h-5" />
                  <span>Emergency Stop Confirmation</span>
                </DialogTitle>
                <DialogDescription>
                  This will immediately halt all trading activities and cancel open orders.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <Label>Reason for Emergency Stop</Label>
                  <Select value={emergencyStopReason} onValueChange={setEmergencyStopReason}>
                    <SelectTrigger className="mt-1">
                      <SelectValue placeholder="Select reason" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="manual">Manual Override</SelectItem>
                      <SelectItem value="risk_breach">Risk Limit Breach</SelectItem>
                      <SelectItem value="technical_issue">Technical Issue</SelectItem>
                      <SelectItem value="market_conditions">Market Conditions</SelectItem>
                      <SelectItem value="compliance">Compliance Issue</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Type "STOP" to confirm</Label>
                  <Input 
                    value={confirmationInput}
                    onChange={(e) => setConfirmationInput(e.target.value)}
                    placeholder="STOP"
                    className="mt-1"
                  />
                </div>
              </div>
              <DialogFooter>
                <Button 
                  variant="destructive"
                  onClick={() => emergencyStopMutation.mutate(emergencyStopReason)}
                  disabled={confirmationInput !== 'STOP' || !emergencyStopReason || emergencyStopMutation.isPending}
                >
                  {emergencyStopMutation.isPending ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      Stopping...
                    </>
                  ) : (
                    <>
                      <Square className="w-4 h-4 mr-2" />
                      EMERGENCY STOP
                    </>
                  )}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          {/* Start/Stop Trading */}
          {tradingStatus?.is_active ? (
            <Button 
              variant="outline"
              onClick={() => stopTradingMutation.mutate()}
              disabled={stopTradingMutation.isPending}
            >
              {stopTradingMutation.isPending ? (
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Square className="w-4 h-4 mr-2" />
              )}
              Stop Trading
            </Button>
          ) : (
            <Button 
              className="bg-gradient-to-r from-green-500 to-emerald-500"
              onClick={() => startTradingMutation.mutate(isLiveModeEnabled ? 'live' : 'paper')}
              disabled={startTradingMutation.isPending}
            >
              {startTradingMutation.isPending ? (
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Play className="w-4 h-4 mr-2" />
              )}
              Start Trading
            </Button>
          )}
        </div>
      </div>

      {/* Status Alerts */}
      {tradingStatus?.mode === 'live' && (
        <Alert className="border-red-600 bg-red-600/10">
          <AlertTriangle className="h-4 w-4 text-red-400" />
          <AlertTitle className="text-red-400">LIVE TRADING MODE ACTIVE</AlertTitle>
          <AlertDescription className="text-red-400/80">
            Real money is at risk. All trades will execute with actual funds. Monitor positions carefully.
          </AlertDescription>
        </Alert>
      )}

      {riskLimits?.some(limit => limit.status === 'breach') && (
        <Alert className="border-red-600 bg-red-600/10">
          <XCircle className="h-4 w-4 text-red-400" />
          <AlertTitle className="text-red-400">Risk Limit Breach Detected</AlertTitle>
          <AlertDescription className="text-red-400/80">
            One or more risk limits have been breached. Review positions immediately.
          </AlertDescription>
        </Alert>
      )}

      {/* Trading Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="glow-hover">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">
              Trading Status
            </CardTitle>
            {tradingStatus?.is_active ? (
              <Activity className="h-4 w-4 text-green-400 animate-pulse" />
            ) : (
              <Pause className="h-4 w-4 text-gray-400" />
            )}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${tradingStatus?.is_active ? 'text-green-400' : 'text-gray-400'}`}>
              {tradingStatus?.is_active ? 'ACTIVE' : 'STOPPED'}
            </div>
            <p className="text-xs text-gray-400 mt-1">
              {tradingStatus?.is_active ? formatUptime(tradingStatus.uptime_seconds) : 'Ready to start'}
            </p>
          </CardContent>
        </Card>

        <Card className="glow-hover">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">
              Daily P&L
            </CardTitle>
            {tradingStatus && tradingStatus.daily_pnl >= 0 ? (
              <TrendingUp className="h-4 w-4 text-green-400" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-400" />
            )}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${
              tradingStatus && tradingStatus.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              {tradingStatus ? formatCurrency(tradingStatus.daily_pnl, true) : '$0.00'}
            </div>
            <p className="text-xs text-gray-400 mt-1">
              {tradingStatus ? formatPercent(tradingStatus.daily_pnl_pct, true) : '0.00%'}
            </p>
          </CardContent>
        </Card>

        <Card className="glow-hover">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">
              Trades Today
            </CardTitle>
            <BarChart3 className="h-4 w-4 text-ocean-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {tradingStatus?.total_trades_today || 0}
            </div>
            <p className="text-xs text-gray-400 mt-1">
              {tradingStatus?.active_strategies || 0} strategies active
            </p>
          </CardContent>
        </Card>

        <Card className="glow-hover">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">
              Available Capital
            </CardTitle>
            <DollarSign className="h-4 w-4 text-wave-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {tradingStatus ? formatCurrency(tradingStatus.current_capital) : '$0.00'}
            </div>
            <p className="text-xs text-gray-400 mt-1">
              {tradingStatus?.mode.toUpperCase()} mode
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="orders" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="orders">Live Orders</TabsTrigger>
          <TabsTrigger value="positions">Positions</TabsTrigger>
          <TabsTrigger value="risk">Risk Monitoring</TabsTrigger>
          <TabsTrigger value="compliance">Compliance</TabsTrigger>
          <TabsTrigger value="controls">Safety Controls</TabsTrigger>
        </TabsList>

        <TabsContent value="orders" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Active Orders</CardTitle>
              <CardDescription>Real-time order status and execution monitoring</CardDescription>
            </CardHeader>
            <CardContent>
              {liveOrders && liveOrders.length > 0 ? (
                <div className="space-y-4">
                  {liveOrders.map((order) => (
                    <div key={order.order_id} className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className="flex flex-col">
                          <div className="flex items-center space-x-2">
                            <Badge className={getOrderStatusColor(order.status)}>
                              {order.status.replace('_', ' ')}
                            </Badge>
                            <span className="text-sm font-medium text-white">
                              {order.side.toUpperCase()} {order.symbol}
                            </span>
                          </div>
                          <div className="text-xs text-gray-400 mt-1">
                            {order.order_type} • {new Date(order.timestamp).toLocaleTimeString()}
                          </div>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className="text-sm font-medium text-white">
                          {order.filled_quantity}/{order.quantity}
                        </div>
                        {order.price && (
                          <div className="text-xs text-gray-400">
                            @ {formatCurrency(order.price)}
                          </div>
                        )}
                      </div>
                      
                      <div className="text-right">
                        <div className="text-sm font-medium text-white">
                          {formatCurrency(order.estimated_cost)}
                        </div>
                        {order.status === 'partially_filled' && (
                          <Progress 
                            value={(order.filled_quantity / order.quantity) * 100} 
                            className="w-20 h-2 mt-1" 
                          />
                        )}
                      </div>
                      
                      <div className="flex space-x-2">
                        <Button size="sm" variant="outline">
                          <Eye className="w-3 h-3" />
                        </Button>
                        <Button size="sm" variant="outline">
                          <XCircle className="w-3 h-3" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No active orders</p>
                  <p className="text-xs mt-1">Orders will appear here when strategies place trades</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="positions" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Current Positions</CardTitle>
              <CardDescription>Real-time position monitoring and P&L tracking</CardDescription>
            </CardHeader>
            <CardContent>
              {positions && positions.length > 0 ? (
                <div className="space-y-4">
                  {positions.map((position, index) => (
                    <div key={index} className="p-4 bg-white/5 rounded-lg">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          <div>
                            <div className="font-medium text-white">{position.symbol}</div>
                            <div className="text-xs text-gray-400">{position.strategy_id}</div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className={`font-medium ${
                            position.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {formatCurrency(position.unrealized_pnl, true)}
                          </div>
                          <div className={`text-xs ${
                            position.unrealized_pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {formatPercent(position.unrealized_pnl_pct, true)}
                          </div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-4 gap-4 text-sm">
                        <div>
                          <div className="text-gray-400">Quantity</div>
                          <div className="text-white">{position.quantity}</div>
                        </div>
                        <div>
                          <div className="text-gray-400">Avg Price</div>
                          <div className="text-white">{formatCurrency(position.average_price)}</div>
                        </div>
                        <div>
                          <div className="text-gray-400">Market Price</div>
                          <div className="text-white">{formatCurrency(position.market_price)}</div>
                        </div>
                        <div>
                          <div className="text-gray-400">Duration</div>
                          <div className="text-white">{position.duration_hours.toFixed(1)}h</div>
                        </div>
                      </div>
                      
                      <div className="mt-3 flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <span className="text-xs text-gray-400">Risk Score:</span>
                          <Progress value={position.risk_score * 100} className="w-20 h-2" />
                          <span className="text-xs text-white">{(position.risk_score * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex space-x-2">
                          <Button size="sm" variant="outline">
                            <Settings className="w-3 h-3" />
                          </Button>
                          <Button size="sm" variant="outline">
                            <XCircle className="w-3 h-3" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <Target className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No open positions</p>
                  <p className="text-xs mt-1">Positions will appear here when strategies open trades</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="risk" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Risk Limits Monitoring</CardTitle>
              <CardDescription>Real-time risk limit utilization and alerts</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {riskLimits?.map((limit) => (
                  <div key={limit.limit_id} className="p-4 bg-white/5 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <h4 className="font-medium text-white">{limit.name}</h4>
                        <Badge className={getRiskStatusColor(limit.status)}>
                          {limit.status}
                        </Badge>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium text-white">
                          {limit.limit_type === 'concentration' || limit.limit_type === 'drawdown' 
                            ? formatPercent(limit.current_value / 100)
                            : formatCurrency(limit.current_value)
                          } / {limit.limit_type === 'concentration' || limit.limit_type === 'drawdown' 
                            ? formatPercent(limit.limit_value / 100)
                            : formatCurrency(limit.limit_value)
                          }
                        </div>
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-400">Utilization</span>
                        <span className="text-white">{limit.utilization_pct.toFixed(1)}%</span>
                      </div>
                      <Progress 
                        value={limit.utilization_pct} 
                        className={`h-2 ${
                          limit.status === 'breach' ? '[&>div]:bg-red-400' :
                          limit.status === 'warning' ? '[&>div]:bg-yellow-400' : '[&>div]:bg-green-400'
                        }`}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="compliance" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Compliance Monitoring</CardTitle>
              <CardDescription>Regulatory compliance checks and status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {complianceChecks?.map((check) => (
                  <div key={check.check_id} className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                    <div className="flex items-center space-x-3">
                      {check.status === 'passing' ? (
                        <CheckCircle className="w-5 h-5 text-green-400" />
                      ) : (
                        <XCircle className="w-5 h-5 text-red-400" />
                      )}
                      <div>
                        <div className="font-medium text-white">{check.rule_name}</div>
                        <div className="text-xs text-gray-400">{check.description}</div>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <Badge className={getComplianceStatusColor(check.severity, check.status)}>
                        {check.status}
                      </Badge>
                      <div className="text-xs text-gray-400 mt-1">
                        {check.current_value}/{check.limit_value}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="controls" className="space-y-6">
          {/* Safety Controls Configuration */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Shield className="w-5 h-5 text-green-400" />
                  <span>Circuit Breakers</span>
                </CardTitle>
                <CardDescription>Automatic safety controls and limits</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-white">Daily Loss Limit</div>
                      <div className="text-xs text-gray-400">Auto-stop at daily loss threshold</div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-400" />
                      <span className="text-xs text-white">$2,000</span>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-white">Drawdown Protection</div>
                      <div className="text-xs text-gray-400">Stop trading on excessive drawdown</div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-400" />
                      <span className="text-xs text-white">5%</span>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-white">Position Size Limit</div>
                      <div className="text-xs text-gray-400">Maximum single position value</div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-400" />
                      <span className="text-xs text-white">$50,000</span>
                    </div>
                  </div>
                </div>
                
                <div className="pt-4 border-t border-white/10">
                  <Button variant="outline" size="sm" className="w-full">
                    <Settings className="w-4 h-4 mr-2" />
                    Configure Limits
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Bell className="w-5 h-5 text-yellow-400" />
                  <span>Alert Settings</span>
                </CardTitle>
                <CardDescription>Notification and alert configuration</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-white">Real-time Alerts</div>
                      <div className="text-xs text-gray-400">Instant notifications for critical events</div>
                    </div>
                    <CheckCircle className="w-4 h-4 text-green-400" />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-white">Email Notifications</div>
                      <div className="text-xs text-gray-400">Send alerts via email</div>
                    </div>
                    <CheckCircle className="w-4 h-4 text-green-400" />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-white">SMS Alerts</div>
                      <div className="text-xs text-gray-400">Critical alerts via SMS</div>
                    </div>
                    <XCircle className="w-4 h-4 text-gray-400" />
                  </div>
                </div>
                
                <div className="pt-4 border-t border-white/10">
                  <Button variant="outline" size="sm" className="w-full">
                    <Bell className="w-4 h-4 mr-2" />
                    Configure Alerts
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Emergency Protocols */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <AlertTriangle className="w-5 h-5 text-red-400" />
                <span>Emergency Protocols</span>
              </CardTitle>
              <CardDescription>Emergency procedures and recovery options</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-red-400/10 border border-red-400/20 rounded-lg text-center">
                  <AlertTriangle className="w-8 h-8 text-red-400 mx-auto mb-2" />
                  <h4 className="font-medium text-red-400 mb-2">Immediate Stop</h4>
                  <p className="text-sm text-red-400/80 mb-3">
                    Stop all trading immediately and cancel open orders
                  </p>
                  <Button size="sm" variant="destructive" className="w-full">
                    Execute Stop
                  </Button>
                </div>
                
                <div className="p-4 bg-yellow-400/10 border border-yellow-400/20 rounded-lg text-center">
                  <Pause className="w-8 h-8 text-yellow-400 mx-auto mb-2" />
                  <h4 className="font-medium text-yellow-400 mb-2">Pause Trading</h4>
                  <p className="text-sm text-yellow-400/80 mb-3">
                    Pause new trades but maintain current positions
                  </p>
                  <Button size="sm" variant="outline" className="w-full">
                    Pause Trading
                  </Button>
                </div>
                
                <div className="p-4 bg-blue-400/10 border border-blue-400/20 rounded-lg text-center">
                  <SkipForward className="w-8 h-8 text-blue-400 mx-auto mb-2" />
                  <h4 className="font-medium text-blue-400 mb-2">Safe Mode</h4>
                  <p className="text-sm text-blue-400/80 mb-3">
                    Reduce position sizes and increase safety margins
                  </p>
                  <Button size="sm" variant="outline" className="w-full">
                    Enable Safe Mode
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}