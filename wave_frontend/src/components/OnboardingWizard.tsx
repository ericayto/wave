import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { Progress } from './ui/progress'
import { Badge } from './ui/badge'
import { cn } from '../lib/utils'
import { 
  ChevronRight, 
  ChevronLeft, 
  Waves,
  Shield,
  Brain,
  TrendingUp,
  Settings,
  CheckCircle,
  Sparkles,
  Rocket
} from 'lucide-react'

// Step components
import { WelcomeStep } from './wizard/WelcomeStep'
import { ExchangeStep } from './wizard/ExchangeStep'
import { LLMProviderStep } from './wizard/LLMProviderStep'
import { RiskProfileStep } from './wizard/RiskProfileStep'
import { TradingPreferencesStep } from './wizard/TradingPreferencesStep'
import { FinalSetupStep } from './wizard/FinalSetupStep'

export interface WizardData {
  // Exchange configuration
  exchange: {
    krakenApiKey: string
    krakenApiSecret: string
    sandboxMode: boolean
  }
  
  // LLM Provider configuration  
  llmProvider: {
    provider: 'openai' | 'azure' | 'openrouter' | 'local'
    apiKey: string
    endpoint?: string
    model: string
    hourlyTokenBudget: number
    dailyTokenBudget: number
  }
  
  // Risk configuration
  risk: {
    profile: 'conservative' | 'moderate' | 'aggressive' | 'custom'
    maxPositionPct: number
    dailyLossLimitPct: number
    maxOrdersPerHour: number
    circuitBreakerSpreadBps: number
  }
  
  // Trading preferences
  trading: {
    defaultSymbols: string[]
    defaultTimeframe: string
    maxActiveStrategies: number
    baseCurrency: string
  }
}

const initialWizardData: WizardData = {
  exchange: {
    krakenApiKey: '',
    krakenApiSecret: '',
    sandboxMode: true
  },
  llmProvider: {
    provider: 'openai',
    apiKey: '',
    model: 'gpt-4o-mini',
    hourlyTokenBudget: 50000,
    dailyTokenBudget: 500000
  },
  risk: {
    profile: 'moderate',
    maxPositionPct: 0.25,
    dailyLossLimitPct: 2.0,
    maxOrdersPerHour: 6,
    circuitBreakerSpreadBps: 50
  },
  trading: {
    defaultSymbols: ['BTC/USDT', 'ETH/USDT'],
    defaultTimeframe: '5m',
    maxActiveStrategies: 5,
    baseCurrency: 'USD'
  }
}

const steps = [
  {
    id: 'welcome',
    title: 'Welcome',
    description: 'Getting started with Wave',
    icon: Waves,
    component: WelcomeStep
  },
  {
    id: 'exchange',
    title: 'Exchange Setup',
    description: 'Connect your trading account',
    icon: Shield,
    component: ExchangeStep
  },
  {
    id: 'llm',
    title: 'AI Provider',
    description: 'Configure your LLM provider',
    icon: Brain,
    component: LLMProviderStep
  },
  {
    id: 'risk',
    title: 'Risk Profile',
    description: 'Set your risk parameters',
    icon: TrendingUp,
    component: RiskProfileStep
  },
  {
    id: 'preferences',
    title: 'Trading Setup',
    description: 'Configure trading preferences',
    icon: Settings,
    component: TradingPreferencesStep
  },
  {
    id: 'finish',
    title: 'Launch',
    description: 'Complete setup and launch bot',
    icon: Rocket,
    component: FinalSetupStep
  }
]

interface OnboardingWizardProps {
  onComplete: (data: WizardData) => void
  onSkip?: () => void
}

export const OnboardingWizard: React.FC<OnboardingWizardProps> = ({ 
  onComplete, 
  onSkip 
}) => {
  const [currentStep, setCurrentStep] = useState(0)
  const [wizardData, setWizardData] = useState<WizardData>(initialWizardData)
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set())
  const [isTransitioning, setIsTransitioning] = useState(false)

  const currentStepData = steps[currentStep]
  const StepComponent = currentStepData.component
  const progress = ((currentStep) / (steps.length - 1)) * 100

  const updateWizardData = (updates: Partial<WizardData>) => {
    setWizardData(prev => ({
      ...prev,
      ...updates
    }))
  }

  const validateStep = (stepIndex: number): boolean => {
    const step = steps[stepIndex]
    
    switch (step.id) {
      case 'welcome':
        return true
      case 'exchange':
        return wizardData.exchange.krakenApiKey.length > 0 && 
               wizardData.exchange.krakenApiSecret.length > 0
      case 'llm':
        return wizardData.llmProvider.apiKey.length > 0 || 
               wizardData.llmProvider.provider === 'local'
      case 'risk':
        return wizardData.risk.maxPositionPct > 0 && 
               wizardData.risk.dailyLossLimitPct > 0
      case 'preferences':
        return wizardData.trading.defaultSymbols.length > 0
      case 'finish':
        return true
      default:
        return false
    }
  }

  const goToStep = async (stepIndex: number) => {
    if (stepIndex < 0 || stepIndex >= steps.length) return
    
    setIsTransitioning(true)
    
    // Add a smooth transition delay
    await new Promise(resolve => setTimeout(resolve, 200))
    
    setCurrentStep(stepIndex)
    setIsTransitioning(false)
  }

  const nextStep = async () => {
    if (currentStep < steps.length - 1) {
      if (validateStep(currentStep)) {
        setCompletedSteps(prev => new Set([...prev, currentStep]))
        await goToStep(currentStep + 1)
      }
    } else {
      // Final step - complete wizard
      handleComplete()
    }
  }

  const prevStep = async () => {
    if (currentStep > 0) {
      await goToStep(currentStep - 1)
    }
  }

  const handleComplete = () => {
    setCompletedSteps(prev => new Set([...prev, currentStep]))
    onComplete(wizardData)
  }

  const isStepValid = validateStep(currentStep)
  const canProceed = isStepValid && currentStep < steps.length - 1

  return (
    <div className="min-h-screen flex items-center justify-center p-4 relative">
      {/* Background Effect */}
      <div className="absolute inset-0 bg-gradient-to-br from-accent-cyan/5 via-transparent to-accent-purple/5 pointer-events-none" />
      
      <div className="w-full max-w-4xl relative z-10">
        {/* Header with Progress */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <div className="w-12 h-12 bg-accent-cyan rounded-glass flex items-center justify-center glow-cyan mr-3">
              <Waves className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-fg-primary text-glow-cyan">Wave Setup</h1>
              <p className="text-sm text-fg-secondary">Let's get your trading bot configured</p>
            </div>
          </div>
          
          <div className="max-w-md mx-auto mb-6">
            <Progress value={progress} className="mb-2" />
            <p className="text-xs text-fg-secondary">
              Step {currentStep + 1} of {steps.length}
            </p>
          </div>
        </div>

        {/* Step Navigation */}
        <div className="flex justify-center mb-8 overflow-x-auto">
          <div className="flex space-x-4 px-4">
            {steps.map((step, index) => {
              const Icon = step.icon
              const isActive = index === currentStep
              const isCompleted = completedSteps.has(index)
              const isAccessible = index <= currentStep || isCompleted

              return (
                <button
                  key={step.id}
                  onClick={() => isAccessible && goToStep(index)}
                  disabled={!isAccessible}
                  className={cn(
                    "flex flex-col items-center space-y-2 p-3 rounded-glass transition-all duration-micro min-w-[100px]",
                    isActive 
                      ? "glass-elev-2 glow-cyan border-accent-cyan/20" 
                      : isCompleted 
                        ? "glass-elev-1 hover:glass-elev-2 border-accent-emerald/20 text-accent-emerald"
                        : isAccessible
                          ? "glass-elev-1 hover:glass-elev-2 text-fg-secondary hover:text-fg-primary"
                          : "glass-elev-1 opacity-50 cursor-not-allowed"
                  )}
                >
                  <div className={cn(
                    "w-8 h-8 rounded-glass flex items-center justify-center transition-all",
                    isActive 
                      ? "bg-accent-cyan text-white" 
                      : isCompleted
                        ? "bg-accent-emerald text-white"
                        : "glass-elev-1"
                  )}>
                    {isCompleted ? (
                      <CheckCircle className="w-4 h-4" />
                    ) : (
                      <Icon className="w-4 h-4" />
                    )}
                  </div>
                  <div className="text-center">
                    <p className={cn(
                      "text-xs font-medium",
                      isActive ? "text-accent-cyan" : isCompleted ? "text-accent-emerald" : ""
                    )}>
                      {step.title}
                    </p>
                  </div>
                </button>
              )
            })}
          </div>
        </div>

        {/* Main Step Content */}
        <Card className={cn(
          "glass-elev-2 transition-all duration-300",
          isTransitioning ? "opacity-50 scale-95" : "opacity-100 scale-100"
        )}>
          <CardHeader className="text-center">
            <div className="flex items-center justify-center mb-4">
              <currentStepData.icon className="w-8 h-8 text-accent-cyan mr-3" />
              <div>
                <CardTitle className="text-xl text-glow-cyan">{currentStepData.title}</CardTitle>
                <CardDescription>{currentStepData.description}</CardDescription>
              </div>
            </div>
          </CardHeader>
          
          <CardContent className="min-h-[400px]">
            <StepComponent 
              data={wizardData}
              onUpdate={updateWizardData}
              isValid={isStepValid}
            />
          </CardContent>

          {/* Navigation Footer */}
          <div className="flex items-center justify-between p-6 border-t border-glass">
            <div className="flex items-center space-x-4">
              {currentStep > 0 && (
                <Button
                  variant="outline"
                  onClick={prevStep}
                  disabled={isTransitioning}
                >
                  <ChevronLeft className="w-4 h-4 mr-2" />
                  Previous
                </Button>
              )}
              
              {onSkip && currentStep === 0 && (
                <Button
                  variant="ghost"
                  onClick={onSkip}
                  className="text-fg-muted hover:text-fg-secondary"
                >
                  Skip Setup
                </Button>
              )}
            </div>

            <div className="flex items-center space-x-4">
              {/* Validation Status */}
              {currentStep > 0 && (
                <Badge variant={isStepValid ? "success" : "warning"}>
                  {isStepValid ? "Complete" : "Incomplete"}
                </Badge>
              )}
              
              <Button
                onClick={nextStep}
                disabled={!isStepValid || isTransitioning}
                className="glow-hover"
              >
                {currentStep === steps.length - 1 ? (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    Complete Setup
                  </>
                ) : (
                  <>
                    Next
                    <ChevronRight className="w-4 h-4 ml-2" />
                  </>
                )}
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}