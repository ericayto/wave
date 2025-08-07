import React, { useState, useEffect } from 'react';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';

interface LLMStatus {
  user_id: number;
  planning_enabled: boolean;
  planning_interval: number;
  current_plan: any;
  context_stats: any;
}

interface UsageStats {
  hourly_usage: Record<string, number>;
  daily_usage: Record<string, number>;
  available_providers: string[];
  cache_size: number;
}

export default function LLMCenter() {
  const [status, setStatus] = useState<LLMStatus | null>(null);
  const [usage, setUsage] = useState<UsageStats | null>(null);
  const [query, setQuery] = useState('');
  const [queryResult, setQueryResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [planningInterval, setPlanningInterval] = useState(300);

  // Fetch LLM status
  const fetchStatus = async () => {
    try {
      const response = await fetch('/api/llm/planner/status/1');
      const data = await response.json();
      if (data.status === 'success') {
        setStatus(data.planner_status);
      }
    } catch (error) {
      console.error('Failed to fetch LLM status:', error);
    }
  };

  // Fetch usage stats
  const fetchUsage = async () => {
    try {
      const response = await fetch('/api/llm/orchestrator/usage');
      const data = await response.json();
      if (data.status === 'success') {
        setUsage(data.usage_stats);
      }
    } catch (error) {
      console.error('Failed to fetch usage stats:', error);
    }
  };

  // Control planner
  const controlPlanner = async (action: string, value?: any) => {
    try {
      const response = await fetch('/api/llm/planner/control', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action,
          value,
          user_id: 1
        }),
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        await fetchStatus();
      }
    } catch (error) {
      console.error('Failed to control planner:', error);
    }
  };

  // Submit query
  const submitQuery = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch('/api/llm/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          user_id: 1
        }),
      });
      
      const data = await response.json();
      setQueryResult(data);
    } catch (error) {
      console.error('Query failed:', error);
      setQueryResult({ status: 'error', error: error.message });
    }
    setLoading(false);
  };

  // Clear cache
  const clearCache = async () => {
    try {
      const response = await fetch('/api/llm/orchestrator/clear-cache', {
        method: 'POST',
      });
      
      if (response.ok) {
        await fetchUsage();
      }
    } catch (error) {
      console.error('Failed to clear cache:', error);
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchUsage();
    
    // Refresh status every 30 seconds
    const interval = setInterval(() => {
      fetchStatus();
      fetchUsage();
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-ocean-100">🧠 LLM Control Center</h1>
        <div className="flex space-x-2">
          <Button onClick={clearCache} variant="outline">
            Clear Cache
          </Button>
        </div>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Planner Status */}
        <Card className="p-6 bg-gradient-to-br from-ocean-800 to-ocean-900 border-ocean-700">
          <h3 className="text-lg font-semibold text-ocean-100 mb-4">Planner Status</h3>
          {status ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-ocean-300">Planning:</span>
                <div className={`px-2 py-1 rounded text-xs ${
                  status.planning_enabled 
                    ? 'bg-green-600 text-white' 
                    : 'bg-red-600 text-white'
                }`}>
                  {status.planning_enabled ? 'Enabled' : 'Disabled'}
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-ocean-300">Interval:</span>
                <span className="text-ocean-100">{status.planning_interval}s</span>
              </div>
              <div className="flex space-x-2 mt-4">
                <Button 
                  size="sm"
                  onClick={() => controlPlanner(status.planning_enabled ? 'disable' : 'enable')}
                  className={status.planning_enabled ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}
                >
                  {status.planning_enabled ? 'Disable' : 'Enable'}
                </Button>
              </div>
            </div>
          ) : (
            <div className="text-ocean-400">Loading...</div>
          )}
        </Card>

        {/* Usage Stats */}
        <Card className="p-6 bg-gradient-to-br from-wave-800 to-wave-900 border-wave-700">
          <h3 className="text-lg font-semibold text-wave-100 mb-4">Token Usage</h3>
          {usage ? (
            <div className="space-y-2">
              <div className="text-sm text-wave-300">Hourly Usage:</div>
              {Object.entries(usage.hourly_usage).map(([provider, tokens]) => (
                <div key={provider} className="flex justify-between">
                  <span className="text-wave-400 capitalize">{provider}:</span>
                  <span className="text-wave-100">{tokens.toLocaleString()}</span>
                </div>
              ))}
              <div className="pt-2 mt-2 border-t border-wave-700">
                <div className="flex justify-between">
                  <span className="text-wave-300">Cache:</span>
                  <span className="text-wave-100">{usage.cache_size}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-wave-300">Providers:</span>
                  <span className="text-wave-100">{usage.available_providers.length}</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-wave-400">Loading...</div>
          )}
        </Card>

        {/* Current Plan */}
        <Card className="p-6 bg-gradient-to-br from-deep-800 to-deep-900 border-deep-700">
          <h3 className="text-lg font-semibold text-deep-100 mb-4">Current Plan</h3>
          {status?.current_plan ? (
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-deep-300">Intent:</span>
                <span className="text-deep-100 capitalize">{status.current_plan.intent}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-deep-300">Status:</span>
                <span className={`capitalize ${
                  status.current_plan.status === 'completed' 
                    ? 'text-green-400' 
                    : status.current_plan.status === 'failed'
                    ? 'text-red-400'
                    : 'text-yellow-400'
                }`}>
                  {status.current_plan.status}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-deep-300">Confidence:</span>
                <span className="text-deep-100">{status.current_plan.confidence}%</span>
              </div>
            </div>
          ) : (
            <div className="text-deep-400">No active plan</div>
          )}
        </Card>
      </div>

      {/* Interactive Query */}
      <Card className="p-6 bg-gradient-to-br from-ocean-800 to-wave-900 border-ocean-700">
        <h3 className="text-xl font-semibold text-ocean-100 mb-4">💬 Query LLM</h3>
        <div className="space-y-4">
          <div>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask me about market conditions, generate strategies, or request analysis..."
              className="w-full h-24 px-3 py-2 bg-ocean-900 border border-ocean-700 rounded-md text-ocean-100 placeholder-ocean-400 focus:outline-none focus:ring-2 focus:ring-ocean-500"
            />
          </div>
          <div className="flex justify-between">
            <Button
              onClick={submitQuery}
              disabled={!query.trim() || loading}
              className="bg-ocean-600 hover:bg-ocean-700"
            >
              {loading ? 'Processing...' : 'Submit Query'}
            </Button>
          </div>
        </div>

        {/* Query Result */}
        {queryResult && (
          <div className="mt-6 p-4 bg-ocean-900 rounded-lg border border-ocean-700">
            <h4 className="text-lg font-medium text-ocean-100 mb-2">Response:</h4>
            <div className="text-ocean-200 whitespace-pre-wrap">
              {queryResult.status === 'success' ? (
                <div>
                  {queryResult.result?.result ? (
                    <div className="space-y-2">
                      <div><strong>Plan ID:</strong> {queryResult.result.result.plan_id}</div>
                      <div><strong>Status:</strong> {queryResult.result.result.status}</div>
                      {queryResult.result.result.results && (
                        <div>
                          <strong>Results:</strong>
                          <pre className="mt-1 text-sm text-ocean-300">
                            {JSON.stringify(queryResult.result.result.results, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  ) : (
                    JSON.stringify(queryResult.result, null, 2)
                  )}
                </div>
              ) : (
                <div className="text-red-400">
                  Error: {queryResult.error || 'Unknown error occurred'}
                </div>
              )}
            </div>
          </div>
        )}
      </Card>

      {/* Planning Interval Control */}
      <Card className="p-6 bg-gradient-to-br from-wave-800 to-deep-900 border-wave-700">
        <h3 className="text-xl font-semibold text-wave-100 mb-4">⚙️ Planning Configuration</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-wave-300 mb-2">
              Planning Interval (seconds)
            </label>
            <div className="flex items-center space-x-4">
              <input
                type="number"
                value={planningInterval}
                onChange={(e) => setPlanningInterval(parseInt(e.target.value) || 300)}
                min={60}
                max={3600}
                className="w-32 px-3 py-2 bg-wave-900 border border-wave-700 rounded-md text-wave-100 focus:outline-none focus:ring-2 focus:ring-wave-500"
              />
              <Button
                onClick={() => controlPlanner('set_interval', planningInterval)}
                className="bg-wave-600 hover:bg-wave-700"
              >
                Update Interval
              </Button>
            </div>
            <p className="text-xs text-wave-400 mt-1">
              How often the LLM reviews market conditions (60-3600 seconds)
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}