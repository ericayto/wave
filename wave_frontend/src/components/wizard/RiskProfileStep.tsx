import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import { WizardData } from '../OnboardingWizard'
import { 
  Shield,
  TrendingUp,
  Zap,
  Settings,
  AlertTriangle,
  Target,
  Gauge,
  DollarSign
} from 'lucide-react'

interface RiskProfileStepProps {
  data: WizardData
  onUpdate: (updates: Partial<WizardData>) => void
  isValid: boolean
}

const riskProfiles = [
  {
    id: 'conservative' as const,
    name: 'Conservative',
    icon: Shield,
    description: 'Play it safe with minimal risk',
    color: 'text-accent-emerald',
    bgColor: 'bg-accent-emerald/10',
    borderColor: 'border-accent-emerald/30',
    settings: {
      maxPositionPct: 0.10,
      dailyLossLimitPct: 1.0,
      maxOrdersPerHour: 3,
      circuitBreakerSpreadBps: 30
    },
    features: [
      'Maximum 10% position size',
      '1% daily loss limit',
      '3 orders per hour max',
      'Tight spread protection'
    ],
    suitableFor: 'Risk-averse traders, beginners'
  },
  {
    id: 'moderate' as const,
    name: 'Moderate',
    icon: TrendingUp,
    description: 'Balanced approach to risk and reward',
    color: 'text-accent-cyan',
    bgColor: 'bg-accent-cyan/10',
    borderColor: 'border-accent-cyan/30',
    settings: {
      maxPositionPct: 0.25,
      dailyLossLimitPct: 2.0,
      maxOrdersPerHour: 6,
      circuitBreakerSpreadBps: 50
    },
    features: [
      'Maximum 25% position size',
      '2% daily loss limit',
      '6 orders per hour max',
      'Standard spread protection'
    ],
    suitableFor: 'Most traders, balanced strategy'
  },
  {
    id: 'aggressive' as const,
    name: 'Aggressive',
    icon: Zap,
    description: 'Higher risk for potentially higher rewards',
    color: 'text-accent-purple',
    bgColor: 'bg-accent-purple/10',
    borderColor: 'border-accent-purple/30',
    settings: {
      maxPositionPct: 0.50,
      dailyLossLimitPct: 5.0,
      maxOrdersPerHour: 12,
      circuitBreakerSpreadBps: 100
    },
    features: [
      'Maximum 50% position size',
      '5% daily loss limit',
      '12 orders per hour max',
      'Relaxed spread protection'
    ],
    suitableFor: 'Experienced traders, high-risk tolerance'
  },
  {
    id: 'custom' as const,
    name: 'Custom',
    icon: Settings,
    description: 'Configure your own risk parameters',
    color: 'text-fg-primary',
    bgColor: 'bg-fg-primary/10',
    borderColor: 'border-fg-primary/30',
    settings: {
      maxPositionPct: 0.25,
      dailyLossLimitPct: 2.0,
      maxOrdersPerHour: 6,
      circuitBreakerSpreadBps: 50
    },
    features: [
      'Fully customizable limits',
      'Advanced risk controls',
      'Tailored to your needs',
      'Expert mode'
    ],
    suitableFor: 'Advanced users, specific requirements'
  }
]

export const RiskProfileStep: React.FC<RiskProfileStepProps> = ({ 
  data, 
  onUpdate, 
  isValid 
}) => {
  const selectedProfile = riskProfiles.find(p => p.id === data.risk.profile)!

  const updateRiskData = (field: keyof WizardData['risk'], value: any) => {
    onUpdate({
      risk: {
        ...data.risk,
        [field]: value
      }
    })
  }

  const selectRiskProfile = (profileId: WizardData['risk']['profile']) => {
    const profile = riskProfiles.find(p => p.id === profileId)!
    
    onUpdate({
      risk: {
        ...data.risk,
        profile: profileId,
        ...profile.settings
      }
    })
  }

  const calculateEstimatedRisk = () => {
    const { maxPositionPct, dailyLossLimitPct } = data.risk
    const worstCase = maxPositionPct * dailyLossLimitPct
    
    if (worstCase <= 0.25) return { level: 'Low', color: 'text-accent-emerald' }
    if (worstCase <= 1.0) return { level: 'Medium', color: 'text-accent-cyan' }
    if (worstCase <= 2.5) return { level: 'High', color: 'text-yellow-400' }
    return { level: 'Very High', color: 'text-red-400' }
  }

  const riskEstimate = calculateEstimatedRisk()

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="w-16 h-16 bg-gradient-to-r from-accent-emerald to-accent-cyan rounded-glass flex items-center justify-center glow-emerald mx-auto">
          <Shield className="w-8 h-8 text-white" />
        </div>
        <h2 className="text-2xl font-bold text-fg-primary">Set Your Risk Profile</h2>
        <p className="text-fg-secondary">
          Choose how aggressively you want your bot to trade
        </p>
      </div>

      {/* Risk Profile Selection */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {riskProfiles.map((profile) => {
          const Icon = profile.icon
          const isSelected = data.risk.profile === profile.id
          
          return (
            <Card 
              key={profile.id}
              className={`cursor-pointer transition-all duration-micro ${
                isSelected 
                  ? `glass-elev-2 ${profile.borderColor} border-2` 
                  : 'glass-elev-1 hover:glass-elev-2 border-border-glass hover:border-accent-cyan/30'
              }`}
              onClick={() => selectRiskProfile(profile.id)}
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`w-10 h-10 rounded-glass flex items-center justify-center ${
                      isSelected ? profile.bgColor : 'glass-elev-2'
                    }`}>
                      <Icon className={`w-5 h-5 ${isSelected ? profile.color : 'text-fg-secondary'}`} />
                    </div>
                    <div>
                      <CardTitle className={`text-lg ${isSelected ? profile.color : ''}`}>
                        {profile.name}
                      </CardTitle>
                      <CardDescription>{profile.description}</CardDescription>
                    </div>
                  </div>
                  <div className={`w-4 h-4 rounded-full border-2 ${
                    isSelected 
                      ? `${profile.color.replace('text-', 'bg-')} ${profile.color.replace('text-', 'border-')}` 
                      : 'border-fg-muted'
                  }`} />
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-3">
                  {/* Features */}
                  <div>
                    <p className="text-xs font-medium text-fg-secondary mb-2">Features:</p>
                    <div className="space-y-1">
                      {profile.features.map((feature, index) => (
                        <div key={index} className="flex items-center space-x-2 text-xs">
                          <div className="w-1 h-1 bg-accent-cyan rounded-full" />
                          <span className="text-fg-secondary">{feature}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  {/* Suitable For */}
                  <div className="pt-2 border-t border-glass">
                    <p className="text-xs text-fg-muted">
                      <strong>Best for:</strong> {profile.suitableFor}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Risk Configuration */}
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-3">
            <selectedProfile.icon className={`w-6 h-6 ${selectedProfile.color}`} />
            <div>
              <CardTitle>{selectedProfile.name} Risk Settings</CardTitle>
              <CardDescription>
                {data.risk.profile === 'custom' 
                  ? 'Customize your risk parameters'
                  : 'Review and adjust if needed'
                }
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Risk Parameters Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-fg-primary flex items-center space-x-2">
                <Target className="w-4 h-4 text-accent-cyan" />
                <span>Maximum Position Size (%)</span>
              </label>
              <Input
                type="number"
                min="1"
                max="100"
                step="1"
                value={data.risk.maxPositionPct * 100}
                onChange={(e) => updateRiskData('maxPositionPct', parseFloat(e.target.value) / 100)}
                disabled={data.risk.profile !== 'custom'}
              />
              <p className="text-xs text-fg-secondary">
                Maximum percentage of portfolio per position
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-fg-primary flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 text-yellow-400" />
                <span>Daily Loss Limit (%)</span>
              </label>
              <Input
                type="number"
                min="0.1"
                max="10"
                step="0.1"
                value={data.risk.dailyLossLimitPct}
                onChange={(e) => updateRiskData('dailyLossLimitPct', parseFloat(e.target.value))}
                disabled={data.risk.profile !== 'custom'}
              />
              <p className="text-xs text-fg-secondary">
                Stop trading after this daily loss percentage
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-fg-primary flex items-center space-x-2">
                <Gauge className="w-4 h-4 text-accent-purple" />
                <span>Max Orders Per Hour</span>
              </label>
              <Input
                type="number"
                min="1"
                max="50"
                step="1"
                value={data.risk.maxOrdersPerHour}
                onChange={(e) => updateRiskData('maxOrdersPerHour', parseInt(e.target.value))}
                disabled={data.risk.profile !== 'custom'}
              />
              <p className="text-xs text-fg-secondary">
                Prevent excessive trading frequency
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-fg-primary flex items-center space-x-2">
                <DollarSign className="w-4 h-4 text-accent-emerald" />
                <span>Circuit Breaker (BPS)</span>
              </label>
              <Input
                type="number"
                min="10"
                max="500"
                step="10"
                value={data.risk.circuitBreakerSpreadBps}
                onChange={(e) => updateRiskData('circuitBreakerSpreadBps', parseInt(e.target.value))}
                disabled={data.risk.profile !== 'custom'}
              />
              <p className="text-xs text-fg-secondary">
                Halt trading if spread exceeds this (basis points)
              </p>
            </div>
          </div>

          {/* Risk Assessment */}
          <div className="pt-4 border-t border-glass">
            <div className="glass-elev-1 rounded-glass p-4">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-medium text-fg-primary">Risk Assessment</h4>
                <Badge variant={
                  riskEstimate.level === 'Low' ? 'success' :
                  riskEstimate.level === 'Medium' ? 'default' :
                  riskEstimate.level === 'High' ? 'warning' : 'destructive'
                }>
                  {riskEstimate.level} Risk
                </Badge>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                <div>
                  <p className="text-xs text-fg-secondary">Max Position</p>
                  <p className="text-sm font-medium text-fg-primary">
                    {(data.risk.maxPositionPct * 100).toFixed(0)}%
                  </p>
                </div>
                <div>
                  <p className="text-xs text-fg-secondary">Daily Limit</p>
                  <p className="text-sm font-medium text-fg-primary">
                    {data.risk.dailyLossLimitPct.toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-xs text-fg-secondary">Orders/Hour</p>
                  <p className="text-sm font-medium text-fg-primary">
                    {data.risk.maxOrdersPerHour}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-fg-secondary">Spread Limit</p>
                  <p className="text-sm font-medium text-fg-primary">
                    {data.risk.circuitBreakerSpreadBps} BPS
                  </p>
                </div>
              </div>

              <div className="mt-4 p-3 glass-elev-1 rounded-glass border border-accent-cyan/20">
                <div className="flex items-start space-x-2">
                  <Shield className="w-4 h-4 text-accent-cyan flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-xs font-medium text-fg-primary mb-1">Risk Protection</p>
                    <p className="text-xs text-fg-secondary">
                      Your settings provide multiple layers of protection: position limits, daily loss caps, 
                      frequency controls, and market condition monitoring.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Warning for High Risk */}
          {riskEstimate.level === 'Very High' && (
            <div className="p-4 glass-elev-1 rounded-glass border border-red-400/20">
              <div className="flex items-start space-x-2">
                <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-red-400 mb-1">High Risk Warning</p>
                  <p className="text-xs text-fg-secondary">
                    Your current settings indicate very high risk levels. Consider reducing position sizes 
                    or daily loss limits, especially when starting with the bot.
                  </p>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Risk Education */}
      <Card className="glass-elev-1">
        <CardHeader>
          <CardTitle className="text-sm">Understanding Risk Parameters</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-fg-secondary">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <p><strong className="text-fg-primary">Position Size:</strong> Controls how much of your portfolio is allocated to each trade</p>
              <p><strong className="text-fg-primary">Daily Loss Limit:</strong> Automatic emergency stop when losses reach this threshold</p>
            </div>
            <div className="space-y-2">
              <p><strong className="text-fg-primary">Order Frequency:</strong> Prevents over-trading and excessive transaction costs</p>
              <p><strong className="text-fg-primary">Spread Protection:</strong> Avoids trading in volatile or illiquid market conditions</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}