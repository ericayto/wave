import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { WizardData } from '../OnboardingWizard'
import { 
  Rocket,
  CheckCircle,
  Shield,
  Brain,
  TrendingUp,
  Settings,
  Sparkles,
  AlertCircle,
  Zap,
  Coins,
  Clock,
  DollarSign,
  Bot
} from 'lucide-react'

interface FinalSetupStepProps {
  data: WizardData
  onUpdate: (updates: Partial<WizardData>) => void
  isValid: boolean
}

export const FinalSetupStep: React.FC<FinalSetupStepProps> = ({ 
  data, 
  onUpdate, 
  isValid 
}) => {
  const [isSaving, setIsSaving] = useState(false)
  const [savedSuccessfully, setSavedSuccessfully] = useState(false)

  const saveConfiguration = async () => {
    setIsSaving(true)
    
    try {
      // Simulate saving configuration
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      // Here you would actually save the configuration to the backend
      console.log('Saving configuration:', data)
      
      setSavedSuccessfully(true)
    } catch (error) {
      console.error('Failed to save configuration:', error)
    } finally {
      setIsSaving(false)
    }
  }

  const configurationSections = [
    {
      icon: Shield,
      title: 'Exchange Setup',
      status: data.exchange.krakenApiKey && data.exchange.krakenApiSecret,
      items: [
        `Exchange: Kraken`,
        `Mode: ${data.exchange.sandboxMode ? 'Paper Trading' : 'Live Trading'}`,
        `API Key: ${data.exchange.krakenApiKey ? '••••••••' : 'Not set'}`
      ]
    },
    {
      icon: Brain,
      title: 'AI Provider',
      status: data.llmProvider.apiKey || data.llmProvider.provider === 'local',
      items: [
        `Provider: ${data.llmProvider.provider.toUpperCase()}`,
        `Model: ${data.llmProvider.model}`,
        `Daily Budget: ${data.llmProvider.dailyTokenBudget.toLocaleString()} tokens`
      ]
    },
    {
      icon: TrendingUp,
      title: 'Risk Profile',
      status: true,
      items: [
        `Profile: ${data.risk.profile.charAt(0).toUpperCase() + data.risk.profile.slice(1)}`,
        `Max Position: ${(data.risk.maxPositionPct * 100).toFixed(0)}%`,
        `Daily Loss Limit: ${data.risk.dailyLossLimitPct}%`
      ]
    },
    {
      icon: Settings,
      title: 'Trading Preferences',
      status: data.trading.defaultSymbols.length > 0,
      items: [
        `Trading Pairs: ${data.trading.defaultSymbols.length} selected`,
        `Timeframe: ${data.trading.defaultTimeframe}`,
        `Base Currency: ${data.trading.baseCurrency}`
      ]
    }
  ]

  const allConfigured = configurationSections.every(section => section.status)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="w-20 h-20 bg-gradient-to-r from-accent-emerald via-accent-cyan to-accent-purple rounded-glass flex items-center justify-center glow-emerald mx-auto">
          {savedSuccessfully ? (
            <CheckCircle className="w-10 h-10 text-white" />
          ) : (
            <Rocket className="w-10 h-10 text-white" />
          )}
        </div>
        <h2 className="text-3xl font-bold text-fg-primary">
          {savedSuccessfully ? 'Setup Complete!' : 'Ready to Launch'}
        </h2>
        <p className="text-lg text-fg-secondary">
          {savedSuccessfully 
            ? 'Your Wave trading bot is now configured and ready to trade'
            : 'Review your configuration and launch your trading bot'
          }
        </p>
      </div>

      {!savedSuccessfully && (
        <>
          {/* Configuration Review */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {configurationSections.map((section, index) => {
              const Icon = section.icon
              
              return (
                <Card 
                  key={index}
                  className={`${
                    section.status 
                      ? 'glass-elev-1 border-accent-emerald/20' 
                      : 'glass-elev-1 border-red-400/20'
                  }`}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <Icon className={`w-5 h-5 ${
                          section.status ? 'text-accent-emerald' : 'text-red-400'
                        }`} />
                        <CardTitle className="text-sm">{section.title}</CardTitle>
                      </div>
                      <Badge variant={section.status ? "success" : "destructive"}>
                        {section.status ? (
                          <CheckCircle className="w-3 h-3 mr-1" />
                        ) : (
                          <AlertCircle className="w-3 h-3 mr-1" />
                        )}
                        {section.status ? 'Ready' : 'Incomplete'}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="space-y-1">
                      {section.items.map((item, itemIndex) => (
                        <div key={itemIndex} className="flex items-center space-x-2 text-xs">
                          <div className="w-1 h-1 bg-accent-cyan rounded-full" />
                          <span className="text-fg-secondary">{item}</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>

          {/* Configuration Summary */}
          <Card className="glass-elev-2 border border-accent-cyan/30">
            <CardHeader>
              <div className="flex items-center space-x-3">
                <Sparkles className="w-6 h-6 text-accent-cyan" />
                <div>
                  <CardTitle className="text-accent-cyan">Configuration Summary</CardTitle>
                  <CardDescription>Your Wave trading bot setup overview</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="w-12 h-12 bg-accent-emerald/20 rounded-glass flex items-center justify-center mx-auto mb-2">
                    <Shield className="w-6 h-6 text-accent-emerald" />
                  </div>
                  <h4 className="font-medium text-fg-primary mb-1">Security First</h4>
                  <p className="text-xs text-fg-secondary">Paper trading mode with secure API management</p>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 bg-accent-cyan/20 rounded-glass flex items-center justify-center mx-auto mb-2">
                    <Brain className="w-6 h-6 text-accent-cyan" />
                  </div>
                  <h4 className="font-medium text-fg-primary mb-1">AI Powered</h4>
                  <p className="text-xs text-fg-secondary">
                    {data.llmProvider.provider.toUpperCase()} with {data.llmProvider.model}
                  </p>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 bg-accent-purple/20 rounded-glass flex items-center justify-center mx-auto mb-2">
                    <TrendingUp className="w-6 h-6 text-accent-purple" />
                  </div>
                  <h4 className="font-medium text-fg-primary mb-1">Risk Managed</h4>
                  <p className="text-xs text-fg-secondary">
                    {data.risk.profile.charAt(0).toUpperCase() + data.risk.profile.slice(1)} risk profile
                  </p>
                </div>
              </div>

              <div className="pt-4 border-t border-glass">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                  <div>
                    <div className="flex items-center justify-center space-x-1 mb-1">
                      <Coins className="w-4 h-4 text-accent-cyan" />
                      <span className="text-xs text-fg-secondary">Pairs</span>
                    </div>
                    <p className="text-sm font-medium text-fg-primary">
                      {data.trading.defaultSymbols.length}
                    </p>
                  </div>
                  <div>
                    <div className="flex items-center justify-center space-x-1 mb-1">
                      <Clock className="w-4 h-4 text-accent-purple" />
                      <span className="text-xs text-fg-secondary">Timeframe</span>
                    </div>
                    <p className="text-sm font-medium text-fg-primary">
                      {data.trading.defaultTimeframe}
                    </p>
                  </div>
                  <div>
                    <div className="flex items-center justify-center space-x-1 mb-1">
                      <DollarSign className="w-4 h-4 text-accent-emerald" />
                      <span className="text-xs text-fg-secondary">Currency</span>
                    </div>
                    <p className="text-sm font-medium text-fg-primary">
                      {data.trading.baseCurrency}
                    </p>
                  </div>
                  <div>
                    <div className="flex items-center justify-center space-x-1 mb-1">
                      <TrendingUp className="w-4 h-4 text-yellow-400" />
                      <span className="text-xs text-fg-secondary">Max Pos.</span>
                    </div>
                    <p className="text-sm font-medium text-fg-primary">
                      {(data.risk.maxPositionPct * 100).toFixed(0)}%
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Save Configuration */}
          {allConfigured && (
            <Card className="glass-elev-1 border border-accent-emerald/20">
              <CardContent className="p-6">
                <div className="text-center space-y-4">
                  <div className="w-16 h-16 bg-gradient-to-r from-accent-emerald to-accent-cyan rounded-glass flex items-center justify-center glow-emerald mx-auto">
                    <Bot className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-fg-primary mb-2">Ready to Save Configuration</h3>
                    <p className="text-fg-secondary">
                      Your Wave trading bot is fully configured and ready to launch. 
                      Click the button below to save your settings and enable bot controls.
                    </p>
                  </div>
                  <Button
                    onClick={saveConfiguration}
                    disabled={isSaving}
                    size="lg"
                    className="glow-hover glow-emerald"
                  >
                    {isSaving ? (
                      <>
                        <Zap className="w-4 h-4 mr-2 animate-spin" />
                        Saving Configuration...
                      </>
                    ) : (
                      <>
                        <Rocket className="w-4 h-4 mr-2" />
                        Save & Launch Bot
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Incomplete Configuration Warning */}
          {!allConfigured && (
            <Card className="glass-elev-1 border border-red-400/20">
              <CardContent className="p-6">
                <div className="flex items-start space-x-3">
                  <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="text-lg font-medium text-red-400 mb-2">Configuration Incomplete</h3>
                    <p className="text-fg-secondary mb-4">
                      Some required configuration steps are not complete. Please go back and finish:
                    </p>
                    <ul className="space-y-1 text-sm text-fg-secondary">
                      {configurationSections
                        .filter(section => !section.status)
                        .map((section, index) => (
                          <li key={index} className="flex items-center space-x-2">
                            <div className="w-1 h-1 bg-red-400 rounded-full" />
                            <span>{section.title}</span>
                          </li>
                        ))
                      }
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}

      {/* Success State */}
      {savedSuccessfully && (
        <div className="space-y-6">
          <Card className="glass-elev-2 border border-accent-emerald/30 glow-emerald">
            <CardContent className="p-8 text-center">
              <div className="w-20 h-20 bg-gradient-to-r from-accent-emerald to-accent-cyan rounded-glass flex items-center justify-center glow-emerald mx-auto mb-4">
                <CheckCircle className="w-10 h-10 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-accent-emerald mb-2">
                🎉 Setup Complete!
              </h3>
              <p className="text-fg-secondary mb-6">
                Your Wave trading bot has been successfully configured and is ready to start trading. 
                You can now access bot controls from the Dashboard.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="glass-elev-1 rounded-glass p-4">
                  <Bot className="w-8 h-8 text-accent-cyan mx-auto mb-2" />
                  <h4 className="font-medium text-fg-primary mb-1">Bot Ready</h4>
                  <p className="text-xs text-fg-secondary">Start/stop controls available</p>
                </div>
                <div className="glass-elev-1 rounded-glass p-4">
                  <Shield className="w-8 h-8 text-accent-emerald mx-auto mb-2" />
                  <h4 className="font-medium text-fg-primary mb-1">Safe Mode</h4>
                  <p className="text-xs text-fg-secondary">Paper trading enabled</p>
                </div>
                <div className="glass-elev-1 rounded-glass p-4">
                  <Settings className="w-8 h-8 text-accent-purple mx-auto mb-2" />
                  <h4 className="font-medium text-fg-primary mb-1">Customizable</h4>
                  <p className="text-xs text-fg-secondary">Modify settings anytime</p>
                </div>
              </div>

              <div className="space-y-3">
                <h4 className="font-medium text-fg-primary">What's Next?</h4>
                <div className="space-y-2 text-sm text-fg-secondary">
                  <div className="flex items-center justify-center space-x-2">
                    <div className="w-6 h-6 rounded-full bg-accent-cyan/20 text-accent-cyan flex items-center justify-center text-xs font-bold">1</div>
                    <span>Go to Dashboard and start your bot</span>
                  </div>
                  <div className="flex items-center justify-center space-x-2">
                    <div className="w-6 h-6 rounded-full bg-accent-cyan/20 text-accent-cyan flex items-center justify-center text-xs font-bold">2</div>
                    <span>Monitor performance and activity</span>
                  </div>
                  <div className="flex items-center justify-center space-x-2">
                    <div className="w-6 h-6 rounded-full bg-accent-cyan/20 text-accent-cyan flex items-center justify-center text-xs font-bold">3</div>
                    <span>Adjust strategies as needed</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}