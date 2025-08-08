import { useState, useEffect } from 'react';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';

interface MemoryStats {
  faiss_index_size: number;
  embedding_cache_size: number;
  budget: {
    max_context_tokens: number;
    target_window_tokens: number;
    rag_top_k: number;
  };
  current_state_age?: number;
}

interface Event {
  id: number;
  timestamp: string;
  type: string;
  payload: any;
  metadata: any;
}

interface Memory {
  id: number;
  timestamp: string;
  type: string;
  payload: any;
  relevance_score: number;
  content: string;
}

export default function MemoryInspector() {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [recentEvents, setRecentEvents] = useState<Event[]>([]);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [pinnedFacts, setPinnedFacts] = useState<Record<string, any>>({});
  const [contextState, setContextState] = useState<any>(null);
  
  // Forms
  const [memoryQuery, setMemoryQuery] = useState('');
  const [newFactKey, setNewFactKey] = useState('');
  const [newFactValue, setNewFactValue] = useState('');
  const [factTTL, setFactTTL] = useState('');
  
  // Fetch memory stats
  const fetchStats = async () => {
    try {
      const response = await fetch('/api/llm/context/stats/1');
      const data = await response.json();
      if (data.status === 'success') {
        setStats(data.stats);
      }
    } catch (error) {
      console.error('Failed to fetch memory stats:', error);
    }
  };

  // Fetch recent events
  const fetchRecentEvents = async () => {
    try {
      const response = await fetch('/api/llm/context/recent/1?n=20');
      const data = await response.json();
      if (data.status === 'success') {
        setRecentEvents(data.events);
      }
    } catch (error) {
      console.error('Failed to fetch recent events:', error);
    }
  };

  // Fetch pinned facts
  const fetchPinnedFacts = async () => {
    try {
      const response = await fetch('/api/llm/context/pinned-facts/1');
      const data = await response.json();
      if (data.status === 'success') {
        setPinnedFacts(data.facts);
      }
    } catch (error) {
      console.error('Failed to fetch pinned facts:', error);
    }
  };

  // Fetch context state
  const fetchContextState = async () => {
    try {
      const response = await fetch('/api/llm/context/state/1');
      const data = await response.json();
      if (data.status === 'success') {
        setContextState(data.state);
      }
    } catch (error) {
      console.error('Failed to fetch context state:', error);
    }
  };

  // Search memories
  const searchMemories = async () => {
    if (!memoryQuery.trim()) return;

    try {
      const response = await fetch('/api/llm/context/retrieve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: memoryQuery,
          k: 10,
          user_id: 1
        }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        setMemories(data.memories);
      }
    } catch (error) {
      console.error('Memory search failed:', error);
    }
  };

  // Pin new fact
  const pinFact = async () => {
    if (!newFactKey.trim() || !newFactValue.trim()) return;

    try {
      let value = newFactValue;
      try {
        // Try to parse as JSON
        value = JSON.parse(newFactValue);
      } catch {
        // Use as string if not valid JSON
      }

      const response = await fetch('/api/llm/context/pin-fact', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          key: newFactKey,
          value: value,
          ttl: factTTL ? parseInt(factTTL) : null,
          user_id: 1
        }),
      });

      const data = await response.json();
      if (data.status === 'success') {
        setNewFactKey('');
        setNewFactValue('');
        setFactTTL('');
        await fetchPinnedFacts();
      }
    } catch (error) {
      console.error('Failed to pin fact:', error);
    }
  };

  // Unpin fact
  const unpinFact = async (key: string) => {
    try {
      const response = await fetch(`/api/llm/context/pin-fact/1/${encodeURIComponent(key)}`, {
        method: 'DELETE',
      });

      const data = await response.json();
      if (data.status === 'success') {
        await fetchPinnedFacts();
      }
    } catch (error) {
      console.error('Failed to unpin fact:', error);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchRecentEvents();
    fetchPinnedFacts();
    fetchContextState();

    // Refresh every 30 seconds
    const interval = setInterval(() => {
      fetchStats();
      fetchRecentEvents();
      fetchContextState();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const getEventTypeColor = (type: string) => {
    switch (type) {
      case 'plan': return 'bg-blue-600';
      case 'action': return 'bg-green-600';
      case 'observation': return 'bg-yellow-600';
      case 'decision': return 'bg-purple-600';
      case 'error': return 'bg-red-600';
      case 'user_input': return 'bg-indigo-600';
      default: return 'bg-gray-600';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-fg-primary">💭 Memory Inspector</h1>
      </div>

      {/* Memory Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats && (
          <>
            <Card className="p-4 glass-elev-2 border-glass">
              <h3 className="font-semibold text-fg-primary mb-2">Index Size</h3>
              <div className="text-2xl font-bold text-fg-secondary">{stats.faiss_index_size}</div>
              <div className="text-xs text-fg-muted">Embeddings</div>
            </Card>

            <Card className="p-4 glass-elev-2 border-glass">
              <h3 className="font-semibold text-fg-primary mb-2">Cache Size</h3>
              <div className="text-2xl font-bold text-fg-secondary">{stats.embedding_cache_size}</div>
              <div className="text-xs text-fg-muted">Cached Vectors</div>
            </Card>

            <Card className="p-4 glass-elev-2 border-glass">
              <h3 className="font-semibold text-fg-primary mb-2">Context Budget</h3>
              <div className="text-lg font-bold text-fg-secondary">
                {(stats.budget.target_window_tokens / 1000).toFixed(0)}K / {(stats.budget.max_context_tokens / 1000).toFixed(0)}K
              </div>
              <div className="text-xs text-fg-muted">Tokens</div>
            </Card>

            <Card className="p-4 glass-elev-2 border-glass">
              <h3 className="font-semibold text-fg-primary mb-2">RAG Top-K</h3>
              <div className="text-2xl font-bold text-fg-secondary">{stats.budget.rag_top_k}</div>
              <div className="text-xs text-fg-muted">Retrieved Memories</div>
            </Card>
          </>
        )}
      </div>

      {/* Memory Search */}
      <Card className="p-6 glass-elev-2 border-glass">
        <h3 className="text-xl font-semibold text-fg-primary mb-4">🔍 Memory Search (RAG)</h3>
        
        <div className="space-y-4">
          <div className="flex space-x-4">
            <input
              type="text"
              value={memoryQuery}
              onChange={(e) => setMemoryQuery(e.target.value)}
              placeholder="Search memories (e.g., 'bitcoin trading decisions')"
              className="flex-1 px-3 py-2 bg-bg-elev-1 border border-glass rounded-md text-fg-primary placeholder-fg-muted focus:outline-none focus:ring-2 focus:ring-accent-cyan"
            />
            <Button onClick={searchMemories} className="bg-accent-cyan hover:bg-accent-cyan/80">
              Search
            </Button>
          </div>

          {memories.length > 0 && (
            <div className="space-y-3 max-h-96 overflow-y-auto">
              <h4 className="font-medium text-fg-secondary">Search Results:</h4>
              {memories.map((memory) => (
                <div key={memory.id} className="p-3 bg-bg-elev-1 border border-glass rounded">
                  <div className="flex items-center justify-between mb-2">
                    <span className={`px-2 py-1 text-xs rounded ${getEventTypeColor(memory.type)} text-white`}>
                      {memory.type}
                    </span>
                    <div className="text-xs text-fg-muted">{formatTimestamp(memory.timestamp)}</div>
                  </div>
                  <div className="text-sm text-fg-secondary mb-2">{memory.content}</div>
                  <div className="text-xs text-fg-muted">
                    Relevance: {(memory.relevance_score * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pinned Facts */}
        <Card className="p-6 glass-elev-2 border-glass">
          <h3 className="text-xl font-semibold text-fg-primary mb-4">📌 Pinned Facts</h3>
          
          {/* Add new fact */}
          <div className="space-y-3 mb-6 p-4 bg-bg-elev-1 border border-glass rounded">
            <h4 className="font-medium text-fg-secondary">Add New Fact</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <input
                type="text"
                value={newFactKey}
                onChange={(e) => setNewFactKey(e.target.value)}
                placeholder="Key"
                className="px-3 py-2 bg-bg-elev-2 border border-glass rounded text-fg-primary placeholder-fg-muted focus:outline-none focus:ring-2 focus:ring-accent-cyan"
              />
              <input
                type="number"
                value={factTTL}
                onChange={(e) => setFactTTL(e.target.value)}
                placeholder="TTL (seconds, optional)"
                className="px-3 py-2 bg-bg-elev-2 border border-glass rounded text-fg-primary placeholder-fg-muted focus:outline-none focus:ring-2 focus:ring-accent-cyan"
              />
            </div>
            <input
              type="text"
              value={newFactValue}
              onChange={(e) => setNewFactValue(e.target.value)}
              placeholder="Value (JSON or string)"
              className="w-full px-3 py-2 bg-bg-elev-2 border border-glass rounded text-fg-primary placeholder-fg-muted focus:outline-none focus:ring-2 focus:ring-accent-cyan"
            />
            <Button onClick={pinFact} size="sm" className="bg-accent-cyan hover:bg-accent-cyan/80">
              Pin Fact
            </Button>
          </div>

          {/* Current facts */}
          <div className="space-y-2">
            {Object.keys(pinnedFacts).length > 0 ? (
              Object.entries(pinnedFacts).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between p-3 bg-bg-elev-1 border border-glass rounded">
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-fg-primary">{key}</div>
                    <div className="text-sm text-fg-secondary truncate">
                      {typeof value === 'string' ? value : JSON.stringify(value)}
                    </div>
                  </div>
                  <Button 
                    onClick={() => unpinFact(key)}
                    size="sm" 
                    variant="outline"
                    className="ml-2 border-red-600 text-red-400 hover:bg-red-600 hover:text-white"
                  >
                    Unpin
                  </Button>
                </div>
              ))
            ) : (
              <div className="text-fg-muted text-center py-4">No pinned facts</div>
            )}
          </div>
        </Card>

        {/* Recent Events */}
        <Card className="p-6 glass-elev-2 border-glass">
          <h3 className="text-xl font-semibold text-fg-primary mb-4">🕐 Recent Events</h3>
          
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {recentEvents.length > 0 ? (
              recentEvents.map((event) => (
                <div key={event.id} className="p-3 bg-bg-elev-1 border border-glass rounded">
                  <div className="flex items-center justify-between mb-2">
                    <span className={`px-2 py-1 text-xs rounded ${getEventTypeColor(event.type)} text-white`}>
                      {event.type}
                    </span>
                    <div className="text-xs text-fg-muted">{formatTimestamp(event.timestamp)}</div>
                  </div>
                  <div className="text-sm text-fg-secondary">
                    <pre className="whitespace-pre-wrap">{JSON.stringify(event.payload, null, 2)}</pre>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-fg-muted text-center py-4">No recent events</div>
            )}
          </div>
        </Card>
      </div>

      {/* Context State */}
      {contextState && (
        <Card className="p-6 glass-elev-2 border-glass">
          <h3 className="text-xl font-semibold text-fg-primary mb-4">🧠 Current Context State</h3>
          
          <div className="bg-bg-elev-1 border border-glass rounded p-4 overflow-x-auto">
            <pre className="text-fg-secondary text-sm">
              {JSON.stringify(contextState, null, 2)}
            </pre>
          </div>
        </Card>
      )}
    </div>
  );
}