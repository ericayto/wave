import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Input } from '../ui/input'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { WizardData } from '../OnboardingWizard'
import { 
  Brain,
  Zap,
  Globe,
  Server,
  Eye,
  EyeOff,
  CheckCircle,
  AlertCircle,
  ExternalLink,
  Cpu,
  Cloud,
  DollarSign
} from 'lucide-react'

interface LLMProviderStepProps {
  data: WizardData
  onUpdate: (updates: Partial<WizardData>) => void
  isValid: boolean
}

const providers = [
  {
    id: 'openai' as const,
    name: 'OpenAI',
    icon: Zap,
    description: 'Industry-leading AI models including GPT-4',
    features: ['GPT-4o mini', 'Fast responses', 'High quality'],
    cost: 'Paid',
    setup: 'API key required',
    recommended: true,
    fields: [
      { key: 'apiKey', label: 'API Key', type: 'password', required: true },
      { key: 'model', label: 'Model', type: 'select', options: ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'], default: 'gpt-4o-mini' }
    ]
  },
  {
    id: 'azure' as const,
    name: 'Azure OpenAI',
    icon: Cloud,
    description: 'Enterprise-grade OpenAI models via Microsoft Azure',
    features: ['Enterprise security', 'Custom deployments', 'SLA guarantees'],
    cost: 'Paid',
    setup: 'API key + endpoint required',
    recommended: false,
    fields: [
      { key: 'apiKey', label: 'API Key', type: 'password', required: true },
      { key: 'endpoint', label: 'Endpoint URL', type: 'text', required: true },
      { key: 'model', label: 'Deployment Name', type: 'text', default: 'gpt-4o-mini' }
    ]
  },
  {
    id: 'openrouter' as const,
    name: 'OpenRouter',
    icon: Globe,
    description: 'Access to multiple AI models through one API',
    features: ['Multiple models', 'Competitive pricing', 'Easy switching'],
    cost: 'Paid (Lower cost)',
    setup: 'API key required',
    recommended: false,
    fields: [
      { key: 'apiKey', label: 'API Key', type: 'password', required: true },
      { key: 'model', label: 'Model', type: 'select', options: ['meta-llama/llama-3.1-8b-instruct:free', 'anthropic/claude-3-haiku', 'openai/gpt-4o-mini'], default: 'meta-llama/llama-3.1-8b-instruct:free' }
    ]
  },
  {
    id: 'local' as const,
    name: 'Local (Ollama)',
    icon: Server,
    description: 'Run AI models locally on your machine',
    features: ['Privacy focused', 'No API costs', 'Offline capable'],
    cost: 'Free',
    setup: 'Ollama installation required',
    recommended: false,
    fields: [
      { key: 'baseUrl', label: 'Base URL', type: 'text', default: 'http://localhost:11434' },
      { key: 'model', label: 'Model', type: 'select', options: ['llama3.1:8b', 'mistral:7b', 'codellama:7b'], default: 'llama3.1:8b' }
    ]
  }
]

export const LLMProviderStep: React.FC<LLMProviderStepProps> = ({ 
  data, 
  onUpdate, 
  isValid 
}) => {
  const [showApiKey, setShowApiKey] = useState(false)
  const [isTestingConnection, setIsTestingConnection] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'success' | 'error'>('idle')

  const selectedProvider = providers.find(p => p.id === data.llmProvider.provider)!

  const updateLLMData = (field: keyof WizardData['llmProvider'], value: any) => {
    onUpdate({
      llmProvider: {
        ...data.llmProvider,
        [field]: value
      }
    })
  }

  const selectProvider = (providerId: WizardData['llmProvider']['provider']) => {
    const provider = providers.find(p => p.id === providerId)!
    const defaults: any = {
      provider: providerId,
      apiKey: '',
      model: provider.fields.find(f => f.key === 'model')?.default || 'gpt-4o-mini'
    }

    // Set provider-specific defaults
    provider.fields.forEach(field => {
      if (field.default) {
        defaults[field.key] = field.default
      }
    })

    onUpdate({
      llmProvider: {
        ...data.llmProvider,
        ...defaults
      }
    })
  }

  const testConnection = async () => {
    setIsTestingConnection(true)
    
    try {
      // Simulate API test - replace with actual connection test
      await new Promise(resolve => setTimeout(resolve, 2000))
      setConnectionStatus('success')
    } catch (error) {
      setConnectionStatus('error')
    } finally {
      setIsTestingConnection(false)
    }
  }

  const openProviderDocs = (providerId: string) => {
    const urls = {
      openai: 'https://platform.openai.com/docs/api-reference',
      azure: 'https://docs.microsoft.com/en-us/azure/cognitive-services/openai/',
      openrouter: 'https://openrouter.ai/docs',
      local: 'https://ollama.ai/docs'
    }
    window.open(urls[providerId as keyof typeof urls], '_blank')
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="w-16 h-16 bg-gradient-to-r from-accent-purple to-pink-500 rounded-glass flex items-center justify-center glow-purple mx-auto">
          <Brain className="w-8 h-8 text-white" />
        </div>
        <h2 className="text-2xl font-bold text-fg-primary">Choose Your AI Provider</h2>
        <p className="text-fg-secondary">
          Select the AI service that will power your trading decisions
        </p>
      </div>

      {/* Provider Selection */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {providers.map((provider) => {
          const Icon = provider.icon
          const isSelected = data.llmProvider.provider === provider.id
          
          return (
            <Card 
              key={provider.id}
              className={`cursor-pointer transition-all duration-micro ${
                isSelected 
                  ? 'glass-elev-2 border-accent-purple/50 glow-purple' 
                  : 'glass-elev-1 hover:glass-elev-2 border-border-glass hover:border-accent-purple/30'
              }`}
              onClick={() => selectProvider(provider.id)}
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`w-10 h-10 rounded-glass flex items-center justify-center ${
                      isSelected ? 'bg-accent-purple' : 'glass-elev-2'
                    }`}>
                      <Icon className={`w-5 h-5 ${isSelected ? 'text-white' : 'text-accent-purple'}`} />
                    </div>
                    <div>
                      <CardTitle className="text-lg flex items-center space-x-2">
                        <span>{provider.name}</span>
                        {provider.recommended && (
                          <Badge variant="success" className="text-xs">
                            Recommended
                          </Badge>
                        )}
                      </CardTitle>
                      <CardDescription>{provider.description}</CardDescription>
                    </div>
                  </div>
                  <div className={`w-4 h-4 rounded-full border-2 ${
                    isSelected ? 'bg-accent-purple border-accent-purple' : 'border-fg-muted'
                  }`} />
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-3">
                  {/* Features */}
                  <div>
                    <p className="text-xs font-medium text-fg-secondary mb-2">Features:</p>
                    <div className="flex flex-wrap gap-1">
                      {provider.features.map((feature, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          {feature}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  
                  {/* Cost & Setup */}
                  <div className="flex justify-between text-xs">
                    <div className="flex items-center space-x-1">
                      <DollarSign className="w-3 h-3" />
                      <span className="text-fg-secondary">{provider.cost}</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Cpu className="w-3 h-3" />
                      <span className="text-fg-secondary">{provider.setup}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Configuration Form */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <selectedProvider.icon className="w-6 h-6 text-accent-purple" />
              <div>
                <CardTitle>{selectedProvider.name} Configuration</CardTitle>
                <CardDescription>Configure your {selectedProvider.name} settings</CardDescription>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => openProviderDocs(selectedProvider.id)}
              className="text-xs"
            >
              <ExternalLink className="w-3 h-3 mr-1" />
              Docs
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {selectedProvider.fields.map((field) => {
            const value = (data.llmProvider as any)[field.key] || ''
            
            return (
              <div key={field.key} className="space-y-2">
                <label className="text-sm font-medium text-fg-primary">
                  {field.label}
                  {field.required && <span className="text-red-400 ml-1">*</span>}
                </label>
                
                {field.type === 'select' ? (
                  <select
                    value={value}
                    onChange={(e) => updateLLMData(field.key as any, e.target.value)}
                    className="w-full h-10 rounded-glass glass-elev-1 border border-border-glass bg-transparent px-3 py-2 text-sm text-fg-primary focus:outline-none focus:ring-2 focus:ring-accent-purple focus:border-accent-purple/50"
                  >
                    {field.options?.map((option) => (
                      <option key={option} value={option} className="bg-bg-base text-fg-primary">
                        {option}
                      </option>
                    ))}
                  </select>
                ) : (
                  <div className="relative">
                    <Input
                      type={field.type === 'password' ? (showApiKey ? 'text' : 'password') : 'text'}
                      placeholder={`Enter your ${field.label.toLowerCase()}`}
                      value={value}
                      onChange={(e) => updateLLMData(field.key as any, e.target.value)}
                      className="font-mono text-sm pr-10"
                    />
                    {field.type === 'password' && (
                      <button
                        type="button"
                        onClick={() => setShowApiKey(!showApiKey)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-fg-muted hover:text-fg-primary transition-colors"
                      >
                        {showApiKey ? (
                          <EyeOff className="w-4 h-4" />
                        ) : (
                          <Eye className="w-4 h-4" />
                        )}
                      </button>
                    )}
                  </div>
                )}
              </div>
            )
          })}

          {/* Token Budget Settings */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t border-glass">
            <div className="space-y-2">
              <label className="text-sm font-medium text-fg-primary">Hourly Token Budget</label>
              <Input
                type="number"
                min="1000"
                max="1000000"
                value={data.llmProvider.hourlyTokenBudget}
                onChange={(e) => updateLLMData('hourlyTokenBudget', parseInt(e.target.value))}
              />
              <p className="text-xs text-fg-secondary">Maximum tokens to use per hour</p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-fg-primary">Daily Token Budget</label>
              <Input
                type="number"
                min="10000"
                max="10000000"
                value={data.llmProvider.dailyTokenBudget}
                onChange={(e) => updateLLMData('dailyTokenBudget', parseInt(e.target.value))}
              />
              <p className="text-xs text-fg-secondary">Maximum tokens to use per day</p>
            </div>
          </div>

          {/* Connection Test */}
          {((selectedProvider.id !== 'local' && data.llmProvider.apiKey) || 
            (selectedProvider.id === 'local')) && (
            <div className="pt-4 border-t border-glass">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <p className="text-sm font-medium text-fg-primary">Test Connection</p>
                  <p className="text-xs text-fg-secondary">
                    Verify your {selectedProvider.name} configuration
                  </p>
                </div>
                <div className="flex items-center space-x-3">
                  {connectionStatus === 'success' && (
                    <Badge variant="success">
                      <CheckCircle className="w-3 h-3 mr-1" />
                      Connected
                    </Badge>
                  )}
                  {connectionStatus === 'error' && (
                    <Badge variant="destructive">
                      <AlertCircle className="w-3 h-3 mr-1" />
                      Failed
                    </Badge>
                  )}
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={testConnection}
                    disabled={isTestingConnection}
                  >
                    {isTestingConnection ? 'Testing...' : 'Test'}
                  </Button>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Provider-specific instructions */}
      {selectedProvider.id === 'local' && (
        <Card className="glass-elev-1 border border-yellow-400/20">
          <CardHeader>
            <div className="flex items-center space-x-3">
              <Server className="w-5 h-5 text-yellow-400" />
              <CardTitle className="text-yellow-400">Local Setup Required</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-fg-secondary">
            <p>To use local models, you need to install and run Ollama:</p>
            <div className="space-y-2">
              <p><strong>1.</strong> Install Ollama from https://ollama.ai</p>
              <p><strong>2.</strong> Run: <code className="bg-bg-elev-1 px-2 py-1 rounded">ollama pull llama3.1:8b</code></p>
              <p><strong>3.</strong> Start Ollama service</p>
              <p><strong>4.</strong> Test the connection above</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}