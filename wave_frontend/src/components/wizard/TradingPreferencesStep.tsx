import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Input } from '../ui/input'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { WizardData } from '../OnboardingWizard'
import { 
  Settings,
  TrendingUp,
  Clock,
  DollarSign,
  Plus,
  X,
  CheckCircle,
  BarChart3,
  Coins
} from 'lucide-react'

interface TradingPreferencesStepProps {
  data: WizardData
  onUpdate: (updates: Partial<WizardData>) => void
  isValid: boolean
}

const popularSymbols = [
  'BTC/USDT',
  'ETH/USDT', 
  'BNB/USDT',
  'ADA/USDT',
  'SOL/USDT',
  'DOT/USDT',
  'AVAX/USDT',
  'MATIC/USDT'
]

const timeframes = [
  { value: '1m', label: '1 Minute', description: 'Very fast, high frequency' },
  { value: '5m', label: '5 Minutes', description: 'Good balance of speed and stability' },
  { value: '15m', label: '15 Minutes', description: 'Medium-term signals' },
  { value: '1h', label: '1 Hour', description: 'Longer-term trends' },
  { value: '4h', label: '4 Hours', description: 'Position trading' },
  { value: '1d', label: '1 Day', description: 'Long-term analysis' }
]

const baseCurrencies = [
  { value: 'USD', label: 'US Dollar (USD)', description: 'Most common base currency' },
  { value: 'EUR', label: 'Euro (EUR)', description: 'European markets' },
  { value: 'GBP', label: 'British Pound (GBP)', description: 'UK markets' },
  { value: 'USDT', label: 'Tether (USDT)', description: 'Cryptocurrency stable coin' }
]

export const TradingPreferencesStep: React.FC<TradingPreferencesStepProps> = ({ 
  data, 
  onUpdate, 
  isValid 
}) => {
  const [newSymbol, setNewSymbol] = useState('')

  const updateTradingData = (field: keyof WizardData['trading'], value: any) => {
    onUpdate({
      trading: {
        ...data.trading,
        [field]: value
      }
    })
  }

  const addSymbol = (symbol: string) => {
    if (symbol && !data.trading.defaultSymbols.includes(symbol)) {
      updateTradingData('defaultSymbols', [...data.trading.defaultSymbols, symbol])
      setNewSymbol('')
    }
  }

  const removeSymbol = (symbol: string) => {
    updateTradingData('defaultSymbols', data.trading.defaultSymbols.filter(s => s !== symbol))
  }

  const addPopularSymbol = (symbol: string) => {
    addSymbol(symbol)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="w-16 h-16 bg-gradient-to-r from-accent-cyan to-accent-purple rounded-glass flex items-center justify-center glow-cyan mx-auto">
          <BarChart3 className="w-8 h-8 text-white" />
        </div>
        <h2 className="text-2xl font-bold text-fg-primary">Trading Preferences</h2>
        <p className="text-fg-secondary">
          Configure your default trading parameters and preferences
        </p>
      </div>

      {/* Trading Symbols */}
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-3">
            <TrendingUp className="w-6 h-6 text-accent-cyan" />
            <div>
              <CardTitle>Trading Pairs</CardTitle>
              <CardDescription>Select the cryptocurrency pairs you want to trade</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Current Symbols */}
          <div>
            <label className="text-sm font-medium text-fg-primary mb-2 block">
              Selected Trading Pairs
              {data.trading.defaultSymbols.length === 0 && <span className="text-red-400 ml-1">*</span>}
            </label>
            <div className="flex flex-wrap gap-2 mb-3 min-h-[40px] p-3 glass-elev-1 rounded-glass border border-border-glass">
              {data.trading.defaultSymbols.length > 0 ? (
                data.trading.defaultSymbols.map((symbol) => (
                  <Badge 
                    key={symbol} 
                    variant="outline" 
                    className="flex items-center space-x-1 glass-hover"
                  >
                    <Coins className="w-3 h-3" />
                    <span>{symbol}</span>
                    <button
                      onClick={() => removeSymbol(symbol)}
                      className="ml-1 hover:text-red-400 transition-colors"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </Badge>
                ))
              ) : (
                <p className="text-fg-muted text-sm">No trading pairs selected</p>
              )}
            </div>
          </div>

          {/* Add Custom Symbol */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-fg-primary">Add Custom Pair</label>
            <div className="flex space-x-2">
              <Input
                type="text"
                placeholder="e.g., BTC/USDT"
                value={newSymbol}
                onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    addSymbol(newSymbol)
                  }
                }}
                className="flex-1"
              />
              <Button
                variant="outline"
                onClick={() => addSymbol(newSymbol)}
                disabled={!newSymbol}
              >
                <Plus className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* Popular Symbols */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-fg-primary">Popular Pairs</label>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {popularSymbols.map((symbol) => {
                const isAdded = data.trading.defaultSymbols.includes(symbol)
                return (
                  <Button
                    key={symbol}
                    variant={isAdded ? "default" : "outline"}
                    size="sm"
                    onClick={() => isAdded ? removeSymbol(symbol) : addPopularSymbol(symbol)}
                    className="justify-start"
                    disabled={isAdded}
                  >
                    {isAdded ? (
                      <CheckCircle className="w-3 h-3 mr-1" />
                    ) : (
                      <Plus className="w-3 h-3 mr-1" />
                    )}
                    {symbol}
                  </Button>
                )
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Timeframe Selection */}
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-3">
            <Clock className="w-6 h-6 text-accent-purple" />
            <div>
              <CardTitle>Default Timeframe</CardTitle>
              <CardDescription>Choose your preferred analysis timeframe</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {timeframes.map((timeframe) => {
              const isSelected = data.trading.defaultTimeframe === timeframe.value
              
              return (
                <Card
                  key={timeframe.value}
                  className={`cursor-pointer transition-all duration-micro ${
                    isSelected 
                      ? 'glass-elev-2 border-accent-purple/50 glow-purple' 
                      : 'glass-elev-1 hover:glass-elev-2 border-border-glass hover:border-accent-purple/30'
                  }`}
                  onClick={() => updateTradingData('defaultTimeframe', timeframe.value)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium text-fg-primary">{timeframe.label}</h4>
                        <p className="text-xs text-fg-secondary mt-1">{timeframe.description}</p>
                      </div>
                      <div className={`w-4 h-4 rounded-full border-2 ${
                        isSelected ? 'bg-accent-purple border-accent-purple' : 'border-fg-muted'
                      }`} />
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Base Currency */}
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-3">
            <DollarSign className="w-6 h-6 text-accent-emerald" />
            <div>
              <CardTitle>Base Currency</CardTitle>
              <CardDescription>Choose your portfolio's base currency</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {baseCurrencies.map((currency) => {
              const isSelected = data.trading.baseCurrency === currency.value
              
              return (
                <Card
                  key={currency.value}
                  className={`cursor-pointer transition-all duration-micro ${
                    isSelected 
                      ? 'glass-elev-2 border-accent-emerald/50 glow-emerald' 
                      : 'glass-elev-1 hover:glass-elev-2 border-border-glass hover:border-accent-emerald/30'
                  }`}
                  onClick={() => updateTradingData('baseCurrency', currency.value)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium text-fg-primary">{currency.label}</h4>
                        <p className="text-xs text-fg-secondary mt-1">{currency.description}</p>
                      </div>
                      <div className={`w-4 h-4 rounded-full border-2 ${
                        isSelected ? 'bg-accent-emerald border-accent-emerald' : 'border-fg-muted'
                      }`} />
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Advanced Settings */}
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-3">
            <Settings className="w-6 h-6 text-accent-cyan" />
            <div>
              <CardTitle>Advanced Settings</CardTitle>
              <CardDescription>Fine-tune your trading behavior</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-fg-primary">
              Maximum Active Strategies
            </label>
            <Input
              type="number"
              min="1"
              max="20"
              value={data.trading.maxActiveStrategies}
              onChange={(e) => updateTradingData('maxActiveStrategies', parseInt(e.target.value))}
            />
            <p className="text-xs text-fg-secondary">
              How many strategies can run simultaneously
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Configuration Summary */}
      <Card className="glass-elev-1 border border-accent-cyan/20">
        <CardHeader>
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-6 h-6 text-accent-cyan" />
            <CardTitle className="text-accent-cyan">Configuration Summary</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-fg-secondary">Trading Pairs:</span>
                <span className="text-fg-primary">
                  {data.trading.defaultSymbols.length} selected
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-fg-secondary">Timeframe:</span>
                <span className="text-fg-primary">
                  {timeframes.find(t => t.value === data.trading.defaultTimeframe)?.label}
                </span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-fg-secondary">Base Currency:</span>
                <span className="text-fg-primary">{data.trading.baseCurrency}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-fg-secondary">Max Strategies:</span>
                <span className="text-fg-primary">{data.trading.maxActiveStrategies}</span>
              </div>
            </div>
          </div>

          {data.trading.defaultSymbols.length === 0 && (
            <div className="mt-4 p-3 glass-elev-1 rounded-glass border border-red-400/20">
              <div className="flex items-start space-x-2">
                <X className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-red-400 mb-1">Missing Trading Pairs</p>
                  <p className="text-xs text-fg-secondary">
                    Please select at least one trading pair to continue.
                  </p>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}