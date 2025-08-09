import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { WizardData } from '../OnboardingWizard'
import { 
  Bot,
  Shield,
  Brain,
  TrendingUp,
  Sparkles,
  CheckCircle,
  Zap,
  Target
} from 'lucide-react'

interface WelcomeStepProps {
  data: WizardData
  onUpdate: (updates: Partial<WizardData>) => void
  isValid: boolean
}

const features = [
  {
    icon: Bot,
    title: "Automated Trading",
    description: "Let AI handle your trades 24/7 with advanced strategies"
  },
  {
    icon: Shield,
    title: "Risk Management", 
    description: "Built-in safety features protect your capital"
  },
  {
    icon: Brain,
    title: "AI-Powered Intelligence",
    description: "LLM integration for smart decision making"
  },
  {
    icon: TrendingUp,
    title: "Performance Analytics",
    description: "Track and optimize your trading performance"
  }
]

const benefits = [
  "🚀 Get started in under 5 minutes",
  "📈 Paper trading mode for safe learning",
  "🔐 Secure credential management",
  "🎯 Customizable risk profiles",
  "🧠 Multiple AI provider support"
]

export const WelcomeStep: React.FC<WelcomeStepProps> = ({ 
  data, 
  onUpdate, 
  isValid 
}) => {
  return (
    <div className="space-y-6">
      {/* Welcome Header */}
      <div className="text-center space-y-4">
        <div className="w-20 h-20 bg-gradient-to-r from-accent-cyan to-accent-purple rounded-glass flex items-center justify-center glow-cyan mx-auto">
          <Sparkles className="w-10 h-10 text-white" />
        </div>
        
        <div>
          <h2 className="text-3xl font-bold text-fg-primary mb-2">
            Welcome to <span className="text-glow-cyan">Wave</span>
          </h2>
          <p className="text-lg text-fg-secondary max-w-2xl mx-auto">
            Your intelligent trading companion that combines advanced AI with robust risk management 
            to help you navigate the markets with confidence.
          </p>
        </div>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {features.map((feature, index) => {
          const Icon = feature.icon
          return (
            <Card key={index} className="glass-hover hover:glow-cyan transition-all duration-300">
              <CardHeader>
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 rounded-glass glass-elev-2 flex items-center justify-center">
                    <Icon className="w-5 h-5 text-accent-cyan" />
                  </div>
                  <CardTitle className="text-lg">{feature.title}</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-fg-secondary">
                  {feature.description}
                </CardDescription>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Benefits Section */}
      <Card className="glass-elev-1 border border-accent-emerald/20">
        <CardHeader>
          <div className="flex items-center space-x-3">
            <Target className="w-6 h-6 text-accent-emerald" />
            <CardTitle className="text-accent-emerald">What You'll Get</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {benefits.map((benefit, index) => (
              <div key={index} className="flex items-center space-x-3">
                <CheckCircle className="w-4 h-4 text-accent-emerald flex-shrink-0" />
                <span className="text-fg-secondary">{benefit}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Setup Overview */}
      <Card className="glass-elev-1 border border-accent-cyan/20">
        <CardHeader>
          <div className="flex items-center space-x-3">
            <Zap className="w-6 h-6 text-accent-cyan" />
            <CardTitle className="text-accent-cyan">Quick Setup Process</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-fg-secondary mb-4">
            We'll guide you through configuring your trading bot in just a few simple steps:
          </p>
          <div className="space-y-3">
            <div className="flex items-center space-x-3 text-sm">
              <div className="w-6 h-6 rounded-full bg-accent-cyan/20 text-accent-cyan flex items-center justify-center font-bold">1</div>
              <span className="text-fg-secondary">Connect your exchange account (Kraken)</span>
            </div>
            <div className="flex items-center space-x-3 text-sm">
              <div className="w-6 h-6 rounded-full bg-accent-cyan/20 text-accent-cyan flex items-center justify-center font-bold">2</div>
              <span className="text-fg-secondary">Choose your AI provider (OpenAI, Azure, etc.)</span>
            </div>
            <div className="flex items-center space-x-3 text-sm">
              <div className="w-6 h-6 rounded-full bg-accent-cyan/20 text-accent-cyan flex items-center justify-center font-bold">3</div>
              <span className="text-fg-secondary">Set your risk profile and limits</span>
            </div>
            <div className="flex items-center space-x-3 text-sm">
              <div className="w-6 h-6 rounded-full bg-accent-cyan/20 text-accent-cyan flex items-center justify-center font-bold">4</div>
              <span className="text-fg-secondary">Configure trading preferences</span>
            </div>
            <div className="flex items-center space-x-3 text-sm">
              <div className="w-6 h-6 rounded-full bg-accent-cyan/20 text-accent-cyan flex items-center justify-center font-bold">5</div>
              <span className="text-fg-secondary">Launch your bot and start trading!</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Ready to Start */}
      <div className="text-center py-6">
        <div className="glass-elev-1 rounded-glass p-6 border border-accent-purple/20">
          <div className="w-16 h-16 bg-gradient-to-r from-accent-purple to-accent-cyan rounded-glass flex items-center justify-center glow-purple mx-auto mb-4">
            <Bot className="w-8 h-8 text-white" />
          </div>
          <h3 className="text-xl font-bold text-fg-primary mb-2">Ready to Begin?</h3>
          <p className="text-fg-secondary mb-4">
            Click "Next" to start configuring your Wave trading bot. 
            The setup process takes less than 5 minutes.
          </p>
          <p className="text-xs text-fg-muted">
            💡 Don't worry - you can always modify these settings later in the Settings page
          </p>
        </div>
      </div>
    </div>
  )
}