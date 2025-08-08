import React, { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Progress } from '../components/ui/progress'
import { Input } from '../components/ui/input'
import { Label } from '../components/ui/label'
import { Textarea } from '../components/ui/textarea'
import { formatPercent } from '../lib/utils'
import { 
  Zap, 
  Settings, 
  BarChart3, 
  Play,
  Square,
  RefreshCw,
  Target,
  Brain,
  FlaskConical,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Clock,
  Cpu,
  GitBranch,
  Shuffle
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

interface OptimizationParameter {
  name: string
  param_type: 'real' | 'integer' | 'categorical'
  current_value: any
  bounds: [any, any]
  importance: number
}

interface OptimizationResult {
  strategy_id: string
  optimization_method: string
  best_parameters: Record<string, any>
  best_fitness: number
  best_objectives: Record<string, number>
  total_generations: number
  total_evaluations: number
  convergence_generation: number
  optimization_time: number
  parameter_sensitivity: Record<string, number>
  robustness_score: number
  overfitting_risk: number
}

interface OptimizationStatus {
  status: 'idle' | 'running' | 'completed' | 'failed'
  progress: number
  generation: number
  max_generations: number
  best_fitness: number
  avg_fitness: number
  current_operation: string
}

interface ABTestResult {
  test_id: string
  strategy_a: {
    name: string
    performance: number
    sharpe_ratio: number
    max_drawdown: number
  }
  strategy_b: {
    name: string
    performance: number
    sharpe_ratio: number
    max_drawdown: number
  }
  statistical_significance: number
  confidence_level: number
  winner: 'A' | 'B' | 'inconclusive'
  recommendation: string
}

interface StrategyTemplate {
  id: string
  name: string
  description: string
  parameters: OptimizationParameter[]
  performance_metrics?: {
    sharpe_ratio: number
    max_drawdown: number
    win_rate: number
  }
}

// Mock data and API functions
const mockStrategies: StrategyTemplate[] = [
  {
    id: 'momentum_1',
    name: 'Momentum Strategy',
    description: 'Trend-following strategy with dynamic position sizing',
    parameters: [
      { name: 'lookback_period', param_type: 'integer', current_value: 14, bounds: [5, 50], importance: 0.8 },
      { name: 'entry_threshold', param_type: 'real', current_value: 0.02, bounds: [0.01, 0.05], importance: 0.9 },
      { name: 'stop_loss', param_type: 'real', current_value: 0.03, bounds: [0.01, 0.08], importance: 0.7 },
      { name: 'position_size', param_type: 'real', current_value: 0.1, bounds: [0.05, 0.25], importance: 0.6 }
    ],
    performance_metrics: { sharpe_ratio: 1.2, max_drawdown: -0.08, win_rate: 0.58 }
  },
  {
    id: 'mean_reversion_1',
    name: 'Mean Reversion Strategy',
    description: 'Statistical arbitrage with volatility filtering',
    parameters: [
      { name: 'z_score_entry', param_type: 'real', current_value: 2.0, bounds: [1.5, 3.0], importance: 0.9 },
      { name: 'z_score_exit', param_type: 'real', current_value: 0.5, bounds: [0.1, 1.0], importance: 0.7 },
      { name: 'volatility_filter', param_type: 'real', current_value: 0.15, bounds: [0.1, 0.3], importance: 0.5 },
      { name: 'rebalance_frequency', param_type: 'categorical', current_value: 'daily', bounds: [['hourly', 'daily', 'weekly']], importance: 0.4 }
    ],
    performance_metrics: { sharpe_ratio: 0.9, max_drawdown: -0.05, win_rate: 0.62 }
  }
]

const fetchOptimizationStatus = async (optimizationId?: string): Promise<OptimizationStatus> => {
  await new Promise(resolve => setTimeout(resolve, 300))
  
  if (!optimizationId) {
    return {
      status: 'idle',
      progress: 0,
      generation: 0,
      max_generations: 0,
      best_fitness: 0,
      avg_fitness: 0,
      current_operation: 'Ready to start'
    }
  }
  
  return {
    status: 'running',
    progress: 0.45,
    generation: 23,
    max_generations: 50,
    best_fitness: 1.34,
    avg_fitness: 0.87,
    current_operation: 'Evaluating generation 23'
  }
}

const fetchOptimizationResults = async (): Promise<OptimizationResult[]> => {
  await new Promise(resolve => setTimeout(resolve, 500))
  
  return [
    {
      strategy_id: 'momentum_1',
      optimization_method: 'genetic_algorithm',
      best_parameters: {
        lookback_period: 18,
        entry_threshold: 0.025,
        stop_loss: 0.035,
        position_size: 0.15
      },
      best_fitness: 1.67,
      best_objectives: {
        sharpe_ratio: 1.67,
        max_drawdown: -0.045,
        win_rate: 0.64
      },
      total_generations: 47,
      total_evaluations: 2350,
      convergence_generation: 42,
      optimization_time: 284.5,
      parameter_sensitivity: {
        entry_threshold: 0.87,
        lookback_period: 0.72,
        stop_loss: 0.58,
        position_size: 0.34
      },
      robustness_score: 0.78,
      overfitting_risk: 0.23
    }
  ]
}

const fetchABTestResults = async (): Promise<ABTestResult[]> => {
  await new Promise(resolve => setTimeout(resolve, 400))
  
  return [
    {
      test_id: 'ab_test_001',
      strategy_a: {
        name: 'Original Momentum',
        performance: 0.124,
        sharpe_ratio: 1.2,
        max_drawdown: -0.08
      },
      strategy_b: {
        name: 'Optimized Momentum',
        performance: 0.167,
        sharpe_ratio: 1.67,
        max_drawdown: -0.045
      },
      statistical_significance: 0.95,
      confidence_level: 95,
      winner: 'B',
      recommendation: 'Deploy optimized version - significant improvement in risk-adjusted returns'
    }
  ]
}

export const StrategyOptimization: React.FC = () => {
  const [selectedStrategy, setSelectedStrategy] = useState<string>('')
  const [optimizationMethod, setOptimizationMethod] = useState<string>('genetic_algorithm')
  const [currentOptimizationId, setCurrentOptimizationId] = useState<string>('')
  const [isOptimizationDialogOpen, setIsOptimizationDialogOpen] = useState(false)
  const [optimizationConfig, setOptimizationConfig] = useState({
    generations: 50,
    population_size: 30,
    objectives: ['sharpe_ratio', 'max_drawdown']
  })

  const { data: optimizationStatus, refetch: refetchStatus } = useQuery({
    queryKey: ['optimization-status', currentOptimizationId],
    queryFn: () => fetchOptimizationStatus(currentOptimizationId),
    refetchInterval: currentOptimizationId ? 2000 : false,
  })

  const { data: optimizationResults, isLoading: resultsLoading } = useQuery({
    queryKey: ['optimization-results'],
    queryFn: fetchOptimizationResults,
  })

  const { data: abTestResults, isLoading: abTestLoading } = useQuery({
    queryKey: ['ab-test-results'],
    queryFn: fetchABTestResults,
  })

  const startOptimizationMutation = useMutation({
    mutationFn: async (config: any) => {
      await new Promise(resolve => setTimeout(resolve, 1000))
      return { optimization_id: 'opt_' + Date.now() }
    },
    onSuccess: (data) => {
      setCurrentOptimizationId(data.optimization_id)
      setIsOptimizationDialogOpen(false)
    },
  })

  const handleStartOptimization = () => {
    if (!selectedStrategy) return
    
    const config = {
      strategy_id: selectedStrategy,
      method: optimizationMethod,
      ...optimizationConfig
    }
    
    startOptimizationMutation.mutate(config)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-blue-400 bg-blue-400/10'
      case 'completed': return 'text-green-400 bg-green-400/10'
      case 'failed': return 'text-red-400 bg-red-400/10'
      default: return 'text-gray-400 bg-gray-400/10'
    }
  }

  const getWinnerColor = (winner: string) => {
    switch (winner) {
      case 'A': return 'text-blue-400 bg-blue-400/10'
      case 'B': return 'text-green-400 bg-green-400/10'
      default: return 'text-yellow-400 bg-yellow-400/10'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Strategy Optimization</h1>
          <p className="text-gray-400 mt-2">AI-powered parameter tuning and strategy enhancement</p>
        </div>
        <Dialog open={isOptimizationDialogOpen} onOpenChange={setIsOptimizationDialogOpen}>
          <DialogTrigger asChild>
            <Button className="bg-gradient-to-r from-ocean-500 to-wave-500">
              <Zap className="w-4 h-4 mr-2" />
              Start Optimization
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[600px]">
            <DialogHeader>
              <DialogTitle>Configure Optimization</DialogTitle>
              <DialogDescription>
                Set up parameters for genetic algorithm optimization
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="strategy" className="text-right">
                  Strategy
                </Label>
                <Select value={selectedStrategy} onValueChange={setSelectedStrategy}>
                  <SelectTrigger className="col-span-3">
                    <SelectValue placeholder="Select strategy to optimize" />
                  </SelectTrigger>
                  <SelectContent>
                    {mockStrategies.map((strategy) => (
                      <SelectItem key={strategy.id} value={strategy.id}>
                        {strategy.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="method" className="text-right">
                  Method
                </Label>
                <Select value={optimizationMethod} onValueChange={setOptimizationMethod}>
                  <SelectTrigger className="col-span-3">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="genetic_algorithm">Genetic Algorithm</SelectItem>
                    <SelectItem value="bayesian_optimization">Bayesian Optimization</SelectItem>
                    <SelectItem value="multi_objective">Multi-Objective NSGA-II</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="generations" className="text-right">
                  Generations
                </Label>
                <Input
                  id="generations"
                  type="number"
                  value={optimizationConfig.generations}
                  onChange={(e) => setOptimizationConfig(prev => ({...prev, generations: parseInt(e.target.value)}))}
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="population" className="text-right">
                  Population
                </Label>
                <Input
                  id="population"
                  type="number"
                  value={optimizationConfig.population_size}
                  onChange={(e) => setOptimizationConfig(prev => ({...prev, population_size: parseInt(e.target.value)}))}
                  className="col-span-3"
                />
              </div>
            </div>
            <DialogFooter>
              <Button 
                onClick={handleStartOptimization} 
                disabled={!selectedStrategy || startOptimizationMutation.isPending}
              >
                {startOptimizationMutation.isPending ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Start Optimization
                  </>
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="genetic">Genetic Algorithm</TabsTrigger>
          <TabsTrigger value="bayesian">Bayesian Optimization</TabsTrigger>
          <TabsTrigger value="abtesting">A/B Testing</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Current Optimization Status */}
          {optimizationStatus && optimizationStatus.status !== 'idle' && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Cpu className="w-5 h-5 text-blue-400" />
                  <span>Optimization in Progress</span>
                </CardTitle>
                <CardDescription>Real-time optimization status and metrics</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <Badge className={getStatusColor(optimizationStatus.status)}>
                    {optimizationStatus.status.toUpperCase()}
                  </Badge>
                  <span className="text-sm text-gray-400">
                    Generation {optimizationStatus.generation} of {optimizationStatus.max_generations}
                  </span>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Progress</span>
                    <span className="text-white">{(optimizationStatus.progress * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={optimizationStatus.progress * 100} className="h-2" />
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-gray-400">Best Fitness</div>
                    <div className="text-xl font-bold text-green-400">
                      {optimizationStatus.best_fitness.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-400">Avg Fitness</div>
                    <div className="text-xl font-bold text-white">
                      {optimizationStatus.avg_fitness.toFixed(3)}
                    </div>
                  </div>
                </div>
                
                <div className="pt-2 border-t border-white/10">
                  <p className="text-sm text-gray-400">
                    {optimizationStatus.current_operation}
                  </p>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Strategy Selection Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {mockStrategies.map((strategy) => (
              <Card key={strategy.id} className="glow-hover cursor-pointer" onClick={() => setSelectedStrategy(strategy.id)}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>{strategy.name}</span>
                    {selectedStrategy === strategy.id && (
                      <CheckCircle className="w-5 h-5 text-green-400" />
                    )}
                  </CardTitle>
                  <CardDescription>{strategy.description}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Current Performance */}
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-sm text-gray-400">Sharpe</div>
                      <div className="font-bold text-white">
                        {strategy.performance_metrics?.sharpe_ratio.toFixed(2)}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">Max DD</div>
                      <div className="font-bold text-red-400">
                        {strategy.performance_metrics && formatPercent(strategy.performance_metrics.max_drawdown)}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">Win Rate</div>
                      <div className="font-bold text-green-400">
                        {strategy.performance_metrics && formatPercent(strategy.performance_metrics.win_rate)}
                      </div>
                    </div>
                  </div>

                  {/* Parameters */}
                  <div>
                    <h4 className="font-medium text-white mb-2">Parameters ({strategy.parameters.length})</h4>
                    <div className="grid grid-cols-2 gap-2">
                      {strategy.parameters.slice(0, 4).map((param) => (
                        <div key={param.name} className="text-xs">
                          <span className="text-gray-400">{param.name}:</span>
                          <span className="text-white ml-1">{param.current_value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Optimization Methods */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="glow-hover">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="w-5 h-5 text-ocean-400" />
                  <span>Genetic Algorithm</span>
                </CardTitle>
                <CardDescription>Evolution-based parameter optimization</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-300 mb-4">
                  Uses evolutionary principles to find optimal parameter combinations through generations of improvement.
                </p>
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Best for:</span>
                    <span className="text-white">Complex parameter spaces</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Time:</span>
                    <span className="text-white">10-30 minutes</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="glow-hover">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Target className="w-5 h-5 text-wave-400" />
                  <span>Bayesian Optimization</span>
                </CardTitle>
                <CardDescription>Efficient parameter exploration</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-300 mb-4">
                  Uses probabilistic models to efficiently search parameter space with fewer evaluations.
                </p>
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Best for:</span>
                    <span className="text-white">Expensive evaluations</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Time:</span>
                    <span className="text-white">5-15 minutes</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="glow-hover">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <GitBranch className="w-5 h-5 text-green-400" />
                  <span>Multi-Objective</span>
                </CardTitle>
                <CardDescription>Balance multiple objectives</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-300 mb-4">
                  Optimizes for multiple objectives simultaneously (return vs risk vs drawdown).
                </p>
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Best for:</span>
                    <span className="text-white">Balanced strategies</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Time:</span>
                    <span className="text-white">15-45 minutes</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="genetic" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Genetic Algorithm Configuration</CardTitle>
              <CardDescription>Configure evolutionary optimization parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="ga-generations">Generations</Label>
                    <Input 
                      id="ga-generations" 
                      type="number" 
                      value={optimizationConfig.generations}
                      onChange={(e) => setOptimizationConfig(prev => ({...prev, generations: parseInt(e.target.value)}))}
                      className="mt-1" 
                    />
                    <p className="text-xs text-gray-400 mt-1">Number of evolutionary generations (20-100)</p>
                  </div>
                  
                  <div>
                    <Label htmlFor="ga-population">Population Size</Label>
                    <Input 
                      id="ga-population" 
                      type="number" 
                      value={optimizationConfig.population_size}
                      onChange={(e) => setOptimizationConfig(prev => ({...prev, population_size: parseInt(e.target.value)}))}
                      className="mt-1" 
                    />
                    <p className="text-xs text-gray-400 mt-1">Individuals per generation (20-100)</p>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <Label>Optimization Objectives</Label>
                    <div className="mt-2 space-y-2">
                      {['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor'].map((objective) => (
                        <label key={objective} className="flex items-center space-x-2">
                          <input 
                            type="checkbox" 
                            checked={optimizationConfig.objectives.includes(objective)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setOptimizationConfig(prev => ({
                                  ...prev, 
                                  objectives: [...prev.objectives, objective]
                                }))
                              } else {
                                setOptimizationConfig(prev => ({
                                  ...prev,
                                  objectives: prev.objectives.filter(obj => obj !== objective)
                                }))
                              }
                            }}
                            className="rounded"
                          />
                          <span className="text-sm text-white capitalize">{objective.replace('_', ' ')}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="pt-4 border-t border-white/10">
                <h4 className="font-medium text-white mb-3">Algorithm Parameters</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <Label htmlFor="mutation-rate">Mutation Rate</Label>
                    <Input id="mutation-rate" type="number" step="0.01" defaultValue="0.1" className="mt-1" />
                    <p className="text-xs text-gray-400 mt-1">0.05 - 0.2 recommended</p>
                  </div>
                  <div>
                    <Label htmlFor="crossover-rate">Crossover Rate</Label>
                    <Input id="crossover-rate" type="number" step="0.01" defaultValue="0.8" className="mt-1" />
                    <p className="text-xs text-gray-400 mt-1">0.6 - 0.9 recommended</p>
                  </div>
                  <div>
                    <Label htmlFor="elite-ratio">Elite Ratio</Label>
                    <Input id="elite-ratio" type="number" step="0.01" defaultValue="0.1" className="mt-1" />
                    <p className="text-xs text-gray-400 mt-1">0.05 - 0.2 recommended</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="bayesian" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Bayesian Optimization Configuration</CardTitle>
              <CardDescription>Efficient parameter space exploration using Gaussian processes</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="bo-iterations">Iterations</Label>
                    <Input id="bo-iterations" type="number" defaultValue="100" className="mt-1" />
                    <p className="text-xs text-gray-400 mt-1">Number of evaluations (50-200)</p>
                  </div>
                  
                  <div>
                    <Label htmlFor="bo-initial">Initial Points</Label>
                    <Input id="bo-initial" type="number" defaultValue="10" className="mt-1" />
                    <p className="text-xs text-gray-400 mt-1">Random evaluations before optimization</p>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="acquisition">Acquisition Function</Label>
                    <Select defaultValue="ei">
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="ei">Expected Improvement</SelectItem>
                        <SelectItem value="poi">Probability of Improvement</SelectItem>
                        <SelectItem value="ucb">Upper Confidence Bound</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-gray-400 mt-1">Strategy for selecting next evaluation point</p>
                  </div>
                  
                  <div>
                    <Label htmlFor="kernel">Kernel Type</Label>
                    <Select defaultValue="rbf">
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="rbf">RBF (Radial Basis Function)</SelectItem>
                        <SelectItem value="matern">Matérn</SelectItem>
                        <SelectItem value="rational_quadratic">Rational Quadratic</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-gray-400 mt-1">Gaussian process kernel function</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="abtesting" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>A/B Testing Results</CardTitle>
              <CardDescription>Statistical comparison of strategy variants</CardDescription>
            </CardHeader>
            <CardContent>
              {abTestLoading ? (
                <div className="space-y-4">
                  {Array.from({ length: 2 }).map((_, i) => (
                    <div key={i} className="shimmer h-20 rounded"></div>
                  ))}
                </div>
              ) : (
                <div className="space-y-4">
                  {abTestResults?.map((test) => (
                    <div key={test.test_id} className="p-4 bg-white/5 rounded-lg">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-2">
                          <FlaskConical className="w-5 h-5 text-ocean-400" />
                          <span className="font-medium text-white">Test {test.test_id}</span>
                        </div>
                        <Badge className={getWinnerColor(test.winner)}>
                          Winner: Strategy {test.winner}
                        </Badge>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
                        {/* Strategy A */}
                        <div className="p-3 bg-blue-400/10 border border-blue-400/20 rounded-lg">
                          <h4 className="font-medium text-blue-400 mb-2">Strategy A: {test.strategy_a.name}</h4>
                          <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-400">Performance:</span>
                              <span className="text-white">{formatPercent(test.strategy_a.performance)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">Sharpe Ratio:</span>
                              <span className="text-white">{test.strategy_a.sharpe_ratio.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">Max Drawdown:</span>
                              <span className="text-red-400">{formatPercent(test.strategy_a.max_drawdown)}</span>
                            </div>
                          </div>
                        </div>
                        
                        {/* Strategy B */}
                        <div className="p-3 bg-green-400/10 border border-green-400/20 rounded-lg">
                          <h4 className="font-medium text-green-400 mb-2">Strategy B: {test.strategy_b.name}</h4>
                          <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-400">Performance:</span>
                              <span className="text-white">{formatPercent(test.strategy_b.performance)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">Sharpe Ratio:</span>
                              <span className="text-white">{test.strategy_b.sharpe_ratio.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">Max Drawdown:</span>
                              <span className="text-red-400">{formatPercent(test.strategy_b.max_drawdown)}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center space-x-4">
                          <span className="text-gray-400">Statistical Significance:</span>
                          <Badge className="text-green-400 bg-green-400/10">
                            {formatPercent(test.statistical_significance)}
                          </Badge>
                        </div>
                        <div className="flex items-center space-x-4">
                          <span className="text-gray-400">Confidence Level:</span>
                          <span className="text-white">{test.confidence_level}%</span>
                        </div>
                      </div>
                      
                      <div className="mt-3 pt-3 border-t border-white/10">
                        <p className="text-sm text-gray-300">{test.recommendation}</p>
                      </div>
                    </div>
                  ))}
                  
                  {/* Start New A/B Test */}
                  <Card className="border-dashed border-gray-600">
                    <CardContent className="p-6 text-center">
                      <FlaskConical className="w-8 h-8 text-gray-500 mx-auto mb-2" />
                      <h4 className="font-medium text-white mb-2">Start New A/B Test</h4>
                      <p className="text-sm text-gray-400 mb-4">
                        Compare strategy variants with statistical significance testing
                      </p>
                      <Button variant="outline">
                        <GitBranch className="w-4 h-4 mr-2" />
                        Configure A/B Test
                      </Button>
                    </CardContent>
                  </Card>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="results" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Optimization Results</CardTitle>
              <CardDescription>Historical optimization results and parameter analysis</CardDescription>
            </CardHeader>
            <CardContent>
              {resultsLoading ? (
                <div className="space-y-4">
                  {Array.from({ length: 2 }).map((_, i) => (
                    <div key={i} className="shimmer h-32 rounded"></div>
                  ))}
                </div>
              ) : (
                <div className="space-y-6">
                  {optimizationResults?.map((result) => (
                    <div key={result.strategy_id} className="p-4 bg-white/5 rounded-lg">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-2">
                          <Settings className="w-5 h-5 text-ocean-400" />
                          <span className="font-medium text-white">
                            {mockStrategies.find(s => s.id === result.strategy_id)?.name || result.strategy_id}
                          </span>
                        </div>
                        <Badge className="text-green-400 bg-green-400/10">
                          {result.optimization_method.replace('_', ' ').toUpperCase()}
                        </Badge>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-4">
                        {/* Best Performance */}
                        <div>
                          <h4 className="font-medium text-white mb-2">Best Performance</h4>
                          <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-400">Fitness Score:</span>
                              <span className="text-green-400 font-medium">{result.best_fitness.toFixed(3)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">Sharpe Ratio:</span>
                              <span className="text-white">{result.best_objectives.sharpe_ratio.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">Win Rate:</span>
                              <span className="text-white">{formatPercent(result.best_objectives.win_rate)}</span>
                            </div>
                          </div>
                        </div>
                        
                        {/* Optimization Stats */}
                        <div>
                          <h4 className="font-medium text-white mb-2">Optimization Stats</h4>
                          <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-400">Generations:</span>
                              <span className="text-white">{result.total_generations}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">Evaluations:</span>
                              <span className="text-white">{result.total_evaluations.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">Time:</span>
                              <span className="text-white">{Math.round(result.optimization_time / 60)}m</span>
                            </div>
                          </div>
                        </div>
                        
                        {/* Quality Metrics */}
                        <div>
                          <h4 className="font-medium text-white mb-2">Quality Metrics</h4>
                          <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-400">Robustness:</span>
                              <span className="text-green-400">{formatPercent(result.robustness_score)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">Overfitting Risk:</span>
                              <span className="text-yellow-400">{formatPercent(result.overfitting_risk)}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Best Parameters */}
                      <div className="mb-4">
                        <h4 className="font-medium text-white mb-2">Optimized Parameters</h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                          {Object.entries(result.best_parameters).map(([param, value]) => (
                            <div key={param} className="p-2 bg-white/5 rounded">
                              <div className="text-xs text-gray-400">{param.replace('_', ' ')}</div>
                              <div className="text-sm font-medium text-white">{value}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      {/* Parameter Sensitivity */}
                      <div>
                        <h4 className="font-medium text-white mb-2">Parameter Sensitivity</h4>
                        <div className="space-y-2">
                          {Object.entries(result.parameter_sensitivity).map(([param, sensitivity]) => (
                            <div key={param} className="flex items-center space-x-3">
                              <span className="text-sm text-gray-400 w-20">{param}:</span>
                              <div className="flex-1">
                                <Progress value={sensitivity * 100} className="h-2" />
                              </div>
                              <span className="text-sm text-white w-12">{(sensitivity * 100).toFixed(0)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}