import { useState } from 'react';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';

interface GeneratedStrategy {
  strategy: any;
  validation: {
    valid: boolean;
    errors: string[];
    warnings: string[];
    score: number;
  };
  analysis?: any;
}

export default function StrategyGenerator() {
  const [description, setDescription] = useState('');
  const [constraints, setConstraints] = useState('{}');
  const [generatedStrategy, setGeneratedStrategy] = useState<GeneratedStrategy | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<any>(null);
  const [refinementRequest, setRefinementRequest] = useState('');

  // Generate strategy from description
  const generateStrategy = async () => {
    if (!description.trim()) return;

    setLoading(true);
    try {
      let constraintsObj = {};
      try {
        constraintsObj = JSON.parse(constraints || '{}');
      } catch (e) {
        console.warn('Invalid constraints JSON, using empty object');
      }

      const response = await fetch('/api/llm/generate-strategy', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          description,
          constraints: constraintsObj,
          user_id: 1
        }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        setGeneratedStrategy(data.result);
      } else {
        console.error('Strategy generation failed:', data);
      }
    } catch (error) {
      console.error('Strategy generation error:', error);
    }
    setLoading(false);
  };

  // Refine existing strategy
  const refineStrategy = async () => {
    if (!selectedStrategy || !refinementRequest.trim()) return;

    setLoading(true);
    try {
      const response = await fetch('/api/llm/refine-strategy', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          strategy_definition: selectedStrategy,
          refinement_request: refinementRequest,
          user_id: 1
        }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        setGeneratedStrategy(data.result);
        setRefinementRequest('');
      }
    } catch (error) {
      console.error('Strategy refinement error:', error);
    }
    setLoading(false);
  };

  // Example descriptions
  const examples = [
    "Create a trend-following strategy that buys when the 20-day SMA crosses above the 50-day SMA with high volume confirmation",
    "Build a mean reversion strategy using RSI that buys when oversold (RSI < 30) and sells when overbought (RSI > 70)",
    "Design a breakout strategy that enters positions when price breaks above Bollinger Bands with volume surge",
    "Make a momentum strategy that follows strong price movements with trailing stop-losses"
  ];

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-fg-primary">🎯 Strategy Generator</h1>
      </div>

      {/* Strategy Generation Form */}
      <Card className="p-6 glass-elev-2 border-glass">
        <h3 className="text-xl font-semibold text-fg-primary mb-4">Generate New Strategy</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-fg-secondary mb-2">
              Strategy Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe your trading strategy in natural language..."
              className="w-full h-32 px-3 py-2 bg-bg-elev-1 border border-glass rounded-md text-fg-primary placeholder-fg-muted focus:outline-none focus:ring-2 focus:ring-accent-cyan"
            />
          </div>

          {/* Example Prompts */}
          <div>
            <label className="block text-sm font-medium text-fg-secondary mb-2">
              Example Descriptions (click to use):
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {examples.map((example, idx) => (
                <Button
                  key={idx}
                  variant="outline"
                  size="sm"
                  onClick={() => setDescription(example)}
                  className="text-left h-auto py-2 px-3 text-xs border-glass text-fg-secondary hover:text-fg-primary"
                >
                  {example}
                </Button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-fg-secondary mb-2">
              Constraints (JSON)
            </label>
            <textarea
              value={constraints}
              onChange={(e) => setConstraints(e.target.value)}
              placeholder='{"symbols": ["BTC/USDT"], "max_position_pct": 0.2}'
              className="w-full h-20 px-3 py-2 bg-bg-elev-1 border border-glass rounded-md text-fg-primary placeholder-fg-muted focus:outline-none focus:ring-2 focus:ring-accent-cyan font-mono text-sm"
            />
          </div>

          <div className="flex space-x-4">
            <Button
              onClick={generateStrategy}
              disabled={!description.trim() || loading}
              className="bg-accent-cyan hover:bg-accent-cyan/80"
            >
              {loading ? 'Generating...' : 'Generate Strategy'}
            </Button>
          </div>
        </div>
      </Card>

      {/* Generated Strategy Display */}
      {generatedStrategy && (
        <div className="space-y-6">
          {/* Validation Status */}
          <Card className="p-6 glass-elev-2 border-glass">
            <h3 className="text-xl font-semibold text-fg-primary mb-4">📊 Strategy Validation</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className={`text-2xl font-bold ${getScoreColor(generatedStrategy.validation.score)}`}>
                  {generatedStrategy.validation.score.toFixed(1)}
                </div>
                <div className="text-fg-secondary text-sm">Quality Score</div>
              </div>
              
              <div className="text-center">
                <div className={`text-2xl font-bold ${
                  generatedStrategy.validation.valid ? 'text-green-400' : 'text-red-400'
                }`}>
                  {generatedStrategy.validation.valid ? '✓' : '✗'}
                </div>
                <div className="text-fg-secondary text-sm">Valid</div>
              </div>
              
              <div className="text-center">
                <div className="text-2xl font-bold text-fg-primary">
                  {generatedStrategy.validation.warnings.length}
                </div>
                <div className="text-fg-secondary text-sm">Warnings</div>
              </div>
            </div>

            {generatedStrategy.validation.errors.length > 0 && (
              <div className="mt-4 p-3 bg-red-900 border border-red-700 rounded">
                <h4 className="font-medium text-red-200 mb-2">Errors:</h4>
                <ul className="list-disc list-inside space-y-1">
                  {generatedStrategy.validation.errors.map((error, idx) => (
                    <li key={idx} className="text-red-300 text-sm">{error}</li>
                  ))}
                </ul>
              </div>
            )}

            {generatedStrategy.validation.warnings.length > 0 && (
              <div className="mt-4 p-3 bg-yellow-900 border border-yellow-700 rounded">
                <h4 className="font-medium text-yellow-200 mb-2">Warnings:</h4>
                <ul className="list-disc list-inside space-y-1">
                  {generatedStrategy.validation.warnings.map((warning, idx) => (
                    <li key={idx} className="text-yellow-300 text-sm">{warning}</li>
                  ))}
                </ul>
              </div>
            )}
          </Card>

          {/* Strategy Definition */}
          <Card className="p-6 glass-elev-2 border-glass">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-fg-primary">🔧 Strategy Definition</h3>
              <Button
                onClick={() => setSelectedStrategy(generatedStrategy.strategy)}
                variant="outline"
                size="sm"
                className="border-glass text-fg-secondary hover:text-fg-primary"
              >
                Select for Refinement
              </Button>
            </div>

            <div className="bg-bg-elev-1 border border-glass rounded-lg p-4 overflow-x-auto">
              <pre className="text-fg-secondary text-sm whitespace-pre-wrap">
                {JSON.stringify(generatedStrategy.strategy, null, 2)}
              </pre>
            </div>
          </Card>

          {/* Strategy Summary */}
          <Card className="p-6 glass-elev-2 border-glass">
            <h3 className="text-xl font-semibold text-fg-primary mb-4">📋 Strategy Summary</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-fg-secondary mb-2">Basic Info</h4>
                <div className="space-y-1 text-sm">
                  <div><span className="text-fg-muted">Name:</span> <span className="text-fg-primary">{generatedStrategy.strategy.name}</span></div>
                  <div><span className="text-fg-muted">Version:</span> <span className="text-fg-primary">{generatedStrategy.strategy.version}</span></div>
                  <div><span className="text-fg-muted">Symbols:</span> <span className="text-fg-primary">{generatedStrategy.strategy.instrument_universe?.join(', ')}</span></div>
                  <div><span className="text-fg-muted">Timeframes:</span> <span className="text-fg-primary">{generatedStrategy.strategy.timeframes?.join(', ')}</span></div>
                </div>
              </div>

              <div>
                <h4 className="font-medium text-fg-secondary mb-2">Risk Parameters</h4>
                <div className="space-y-1 text-sm">
                  <div><span className="text-fg-muted">Max Position:</span> <span className="text-fg-primary">{(generatedStrategy.strategy.risk?.max_position_pct * 100).toFixed(1)}%</span></div>
                  <div><span className="text-fg-muted">Daily Loss Limit:</span> <span className="text-fg-primary">{(generatedStrategy.strategy.risk?.daily_loss_limit_pct).toFixed(1)}%</span></div>
                  <div><span className="text-fg-muted">Max Orders/Hour:</span> <span className="text-fg-primary">{generatedStrategy.strategy.risk?.max_orders_per_hour}</span></div>
                </div>
              </div>
            </div>

            <div className="mt-4">
              <h4 className="font-medium text-fg-secondary mb-2">Signals & Logic</h4>
              <div className="text-sm text-fg-secondary">
                <div><strong>Signals:</strong> {generatedStrategy.strategy.signals?.length || 0} configured</div>
                <div><strong>Entry Conditions:</strong> {generatedStrategy.strategy.entries?.length || 0} defined</div>
                <div><strong>Exit Conditions:</strong> {generatedStrategy.strategy.exits?.length || 0} defined</div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Strategy Refinement */}
      {selectedStrategy && (
        <Card className="p-6 glass-elev-2 border-glass">
          <h3 className="text-xl font-semibold text-fg-primary mb-4">✨ Refine Strategy</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-fg-secondary mb-2">
                Selected Strategy: {selectedStrategy.name}
              </label>
              <div className="text-xs text-fg-muted bg-bg-elev-1 p-2 rounded border border-glass">
                {selectedStrategy.description || 'No description available'}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-fg-secondary mb-2">
                Refinement Request
              </label>
              <textarea
                value={refinementRequest}
                onChange={(e) => setRefinementRequest(e.target.value)}
                placeholder="How would you like to modify this strategy? e.g., 'Make it more conservative by reducing position size' or 'Add volume confirmation to entries'"
                className="w-full h-24 px-3 py-2 bg-bg-elev-1 border border-glass rounded-md text-fg-primary placeholder-fg-muted focus:outline-none focus:ring-2 focus:ring-accent-cyan"
              />
            </div>

            <Button
              onClick={refineStrategy}
              disabled={!refinementRequest.trim() || loading}
              className="bg-accent-cyan hover:bg-accent-cyan/80"
            >
              {loading ? 'Refining...' : 'Refine Strategy'}
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
}