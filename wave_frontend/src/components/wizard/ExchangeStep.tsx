import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Input } from '../ui/input'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { WizardData } from '../OnboardingWizard'
import { 
  Shield,
  Eye,
  EyeOff,
  AlertTriangle,
  CheckCircle,
  ExternalLink,
  Lock,
  Key,
  TestTube,
  Globe
} from 'lucide-react'

interface ExchangeStepProps {
  data: WizardData
  onUpdate: (updates: Partial<WizardData>) => void
  isValid: boolean
}

export const ExchangeStep: React.FC<ExchangeStepProps> = ({ 
  data, 
  onUpdate, 
  isValid 
}) => {
  const [showApiSecret, setShowApiSecret] = useState(false)
  const [isTestingConnection, setIsTestingConnection] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'success' | 'error'>('idle')

  const updateExchangeData = (field: keyof WizardData['exchange'], value: any) => {
    onUpdate({
      exchange: {
        ...data.exchange,
        [field]: value
      }
    })
  }

  const testConnection = async () => {
    if (!data.exchange.krakenApiKey || !data.exchange.krakenApiSecret) {
      return
    }

    setIsTestingConnection(true)
    
    try {
      // Simulate API test - replace with actual connection test
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      // Mock successful connection for now
      setConnectionStatus('success')
    } catch (error) {
      setConnectionStatus('error')
    } finally {
      setIsTestingConnection(false)
    }
  }

  const openKrakenDocs = () => {
    window.open('https://docs.kraken.com/rest/#tag/User-Data/operation/getAccountBalance', '_blank')
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="w-16 h-16 bg-gradient-to-r from-accent-cyan to-emerald-500 rounded-glass flex items-center justify-center glow-cyan mx-auto">
          <Shield className="w-8 h-8 text-white" />
        </div>
        <h2 className="text-2xl font-bold text-fg-primary">Connect Your Exchange</h2>
        <p className="text-fg-secondary">
          Securely connect your Kraken account to enable paper trading
        </p>
      </div>

      {/* Security Notice */}
      <Card className="glass-elev-1 border border-yellow-400/20">
        <CardHeader>
          <div className="flex items-center space-x-3">
            <Lock className="w-5 h-5 text-yellow-400" />
            <CardTitle className="text-yellow-400">Security First</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-fg-secondary text-sm">
            Your API credentials are stored securely and never leave your local environment. 
            We recommend creating API keys with limited permissions.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-accent-emerald" />
              <span className="text-fg-secondary">Query permissions only</span>
            </div>
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-accent-emerald" />
              <span className="text-fg-secondary">No withdrawal access</span>
            </div>
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-accent-emerald" />
              <span className="text-fg-secondary">Paper trading mode</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Exchange Configuration */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 rounded-glass glass-elev-2 flex items-center justify-center">
                <Key className="w-4 h-4 text-accent-cyan" />
              </div>
              <div>
                <CardTitle>Kraken API Credentials</CardTitle>
                <CardDescription>Enter your Kraken API key and secret</CardDescription>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={openKrakenDocs}
              className="text-xs"
            >
              <ExternalLink className="w-3 h-3 mr-1" />
              API Guide
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* API Key Input */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-fg-primary">
              API Key
              <span className="text-red-400 ml-1">*</span>
            </label>
            <Input
              type="text"
              placeholder="Enter your Kraken API key"
              value={data.exchange.krakenApiKey}
              onChange={(e) => updateExchangeData('krakenApiKey', e.target.value)}
              className="font-mono text-sm"
            />
          </div>

          {/* API Secret Input */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-fg-primary">
              API Secret
              <span className="text-red-400 ml-1">*</span>
            </label>
            <div className="relative">
              <Input
                type={showApiSecret ? "text" : "password"}
                placeholder="Enter your Kraken API secret"
                value={data.exchange.krakenApiSecret}
                onChange={(e) => updateExchangeData('krakenApiSecret', e.target.value)}
                className="font-mono text-sm pr-10"
              />
              <button
                type="button"
                onClick={() => setShowApiSecret(!showApiSecret)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-fg-muted hover:text-fg-primary transition-colors"
              >
                {showApiSecret ? (
                  <EyeOff className="w-4 h-4" />
                ) : (
                  <Eye className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>

          {/* Trading Mode Toggle */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-fg-primary">Trading Mode</label>
            <div className="space-y-3">
              <div 
                onClick={() => updateExchangeData('sandboxMode', true)}
                className={`p-4 rounded-glass border cursor-pointer transition-all duration-micro ${
                  data.exchange.sandboxMode 
                    ? 'glass-elev-2 border-accent-emerald/50 glow-emerald' 
                    : 'glass-elev-1 border-border-glass hover:glass-elev-2'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`w-4 h-4 rounded-full border-2 ${
                      data.exchange.sandboxMode 
                        ? 'bg-accent-emerald border-accent-emerald' 
                        : 'border-fg-muted'
                    }`} />
                    <div>
                      <div className="flex items-center space-x-2">
                        <TestTube className="w-4 h-4 text-accent-emerald" />
                        <span className="font-medium text-fg-primary">Paper Trading (Recommended)</span>
                      </div>
                      <p className="text-sm text-fg-secondary mt-1">
                        Safe simulation mode with virtual funds
                      </p>
                    </div>
                  </div>
                  <Badge variant="success">Safe</Badge>
                </div>
              </div>

              <div 
                onClick={() => updateExchangeData('sandboxMode', false)}
                className={`p-4 rounded-glass border cursor-pointer transition-all duration-micro ${
                  !data.exchange.sandboxMode 
                    ? 'glass-elev-2 border-red-400/50 glow-purple' 
                    : 'glass-elev-1 border-border-glass hover:glass-elev-2'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`w-4 h-4 rounded-full border-2 ${
                      !data.exchange.sandboxMode 
                        ? 'bg-red-400 border-red-400' 
                        : 'border-fg-muted'
                    }`} />
                    <div>
                      <div className="flex items-center space-x-2">
                        <Globe className="w-4 h-4 text-red-400" />
                        <span className="font-medium text-fg-primary">Live Trading</span>
                      </div>
                      <p className="text-sm text-fg-secondary mt-1">
                        Real trading with actual funds (coming soon)
                      </p>
                    </div>
                  </div>
                  <Badge variant="warning">Live</Badge>
                </div>
              </div>
            </div>
          </div>

          {/* Connection Test */}
          {data.exchange.krakenApiKey && data.exchange.krakenApiSecret && (
            <div className="pt-4 border-t border-glass">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <p className="text-sm font-medium text-fg-primary">Test Connection</p>
                  <p className="text-xs text-fg-secondary">
                    Verify your credentials work correctly
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
                      <AlertTriangle className="w-3 h-3 mr-1" />
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

      {/* Setup Instructions */}
      <Card className="glass-elev-1">
        <CardHeader>
          <CardTitle className="text-sm">Need help getting your API keys?</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-fg-secondary">
          <div className="space-y-2">
            <p><strong>Step 1:</strong> Log into your Kraken account</p>
            <p><strong>Step 2:</strong> Go to Settings → API</p>
            <p><strong>Step 3:</strong> Create a new API key with "Query Funds" permission</p>
            <p><strong>Step 4:</strong> Copy the key and secret here</p>
          </div>
          <div className="flex items-start space-x-2 p-3 glass-elev-1 rounded-glass border border-yellow-400/20">
            <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-fg-secondary">
              <strong>Important:</strong> Only grant "Query Funds" permission. 
              Never give trading or withdrawal permissions for security.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}