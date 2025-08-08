import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Button } from '../components/ui/button';

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
        <h1 className="text-3xl font-bold text-zinc-200">💭 Memory Inspector</h1>
      </div>

      {/* Memory Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats && (
          <>
            <motion.div 
              whileHover={{ y: -2, scale: 1.01 }} 
              whileTap={{ scale: 0.997 }} 
              transition={{ type: "spring", stiffness: 260, damping: 20 }}
            >
              <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] hover:bg-white/10 transition-colors p-4">
                <h3 className="font-semibold text-zinc-200 mb-2">Index Size</h3>
                <div className="text-2xl font-bold text-zinc-400">{stats.faiss_index_size}</div>
                <div className="text-xs text-zinc-500">Embeddings</div>
              </div>
            </motion.div>

            <motion.div 
              whileHover={{ y: -2, scale: 1.01 }} 
              whileTap={{ scale: 0.997 }} 
              transition={{ type: "spring", stiffness: 260, damping: 20 }}
            >
              <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] hover:bg-white/10 transition-colors p-4">
                <h3 className="font-semibold text-zinc-200 mb-2">Cache Size</h3>
                <div className="text-2xl font-bold text-zinc-400">{stats.embedding_cache_size}</div>
                <div className="text-xs text-zinc-500">Cached Vectors</div>
              </div>
            </motion.div>

            <motion.div 
              whileHover={{ y: -2, scale: 1.01 }} 
              whileTap={{ scale: 0.997 }} 
              transition={{ type: "spring", stiffness: 260, damping: 20 }}
            >
              <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] hover:bg-white/10 transition-colors p-4">
                <h3 className="font-semibold text-zinc-200 mb-2">Context Budget</h3>
                <div className="text-lg font-bold text-zinc-400">
                  {(stats.budget.target_window_tokens / 1000).toFixed(0)}K / {(stats.budget.max_context_tokens / 1000).toFixed(0)}K
                </div>
                <div className="text-xs text-zinc-500">Tokens</div>
              </div>
            </motion.div>

            <motion.div 
              whileHover={{ y: -2, scale: 1.01 }} 
              whileTap={{ scale: 0.997 }} 
              transition={{ type: "spring", stiffness: 260, damping: 20 }}
            >
              <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] hover:bg-white/10 transition-colors p-4">
                <h3 className="font-semibold text-zinc-200 mb-2">RAG Top-K</h3>
                <div className="text-2xl font-bold text-zinc-400">{stats.budget.rag_top_k}</div>
                <div className="text-xs text-zinc-500">Retrieved Memories</div>
              </div>
            </motion.div>
          </>
        )}
      </div>

      {/* Memory Search */}
      <motion.div 
        whileHover={{ y: -2, scale: 1.01 }} 
        whileTap={{ scale: 0.997 }} 
        transition={{ type: "spring", stiffness: 260, damping: 20 }}
      >
        <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] hover:bg-white/10 transition-colors p-6">
          <h3 className="text-xl font-semibold text-zinc-200 mb-4">🔍 Memory Search (RAG)</h3>
          
          <div className="space-y-4">
            <div className="flex space-x-4">
              <input
                type="text"
                value={memoryQuery}
                onChange={(e) => setMemoryQuery(e.target.value)}
                placeholder="Search memories (e.g., 'bitcoin trading decisions')"
                className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-md text-zinc-200 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-cyan-400"
              />
              <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                <Button onClick={searchMemories} className="bg-cyan-400 hover:bg-cyan-500 text-zinc-900">
                  Search
                </Button>
              </motion.div>
            </div>

            {memories.length > 0 && (
              <div className="space-y-3 max-h-96 overflow-y-auto">
                <h4 className="font-medium text-zinc-200">Search Results:</h4>
                {memories.map((memory) => (
                  <div key={memory.id} className="p-3 bg-white/5 border border-white/10 rounded">
                    <div className="flex items-center justify-between mb-2">
                      <span className={`px-2 py-1 text-xs rounded ${getEventTypeColor(memory.type)} text-white`}>
                        {memory.type}
                      </span>
                      <div className="text-xs text-zinc-500">{formatTimestamp(memory.timestamp)}</div>
                    </div>
                    <div className="text-sm text-zinc-200 mb-2">{memory.content}</div>
                    <div className="text-xs text-zinc-500">
                      Relevance: {(memory.relevance_score * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pinned Facts */}
        <motion.div 
          whileHover={{ y: -2, scale: 1.01 }} 
          whileTap={{ scale: 0.997 }} 
          transition={{ type: "spring", stiffness: 260, damping: 20 }}
        >
          <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] hover:bg-white/10 transition-colors p-6">
            <h3 className="text-xl font-semibold text-zinc-200 mb-4">📌 Pinned Facts</h3>
            
            {/* Add new fact */}
            <div className="space-y-3 mb-6 p-4 bg-white/5 border border-white/10 rounded">
              <h4 className="font-medium text-zinc-200">Add New Fact</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <input
                  type="text"
                  value={newFactKey}
                  onChange={(e) => setNewFactKey(e.target.value)}
                  placeholder="Key"
                  className="px-3 py-2 bg-white/5 border border-white/10 rounded text-zinc-200 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                />
                <input
                  type="number"
                  value={factTTL}
                  onChange={(e) => setFactTTL(e.target.value)}
                  placeholder="TTL (seconds, optional)"
                  className="px-3 py-2 bg-white/5 border border-white/10 rounded text-zinc-200 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                />
              </div>
              <input
                type="text"
                value={newFactValue}
                onChange={(e) => setNewFactValue(e.target.value)}
                placeholder="Value (JSON or string)"
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded text-zinc-200 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-cyan-400"
              />
              <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                <Button onClick={pinFact} size="sm" className="bg-cyan-400 hover:bg-cyan-500 text-zinc-900">
                  Pin Fact
                </Button>
              </motion.div>
            </div>

            {/* Current facts */}
            <div className="space-y-2">
              {Object.keys(pinnedFacts).length > 0 ? (
                Object.entries(pinnedFacts).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between p-3 bg-white/5 border border-white/10 rounded">
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-zinc-200">{key}</div>
                      <div className="text-sm text-zinc-400 truncate">
                        {typeof value === 'string' ? value : JSON.stringify(value)}
                      </div>
                    </div>
                    <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                      <Button 
                        onClick={() => unpinFact(key)}
                        size="sm" 
                        variant="outline"
                        className="ml-2 border-red-600 text-red-400 hover:bg-red-600 hover:text-white"
                      >
                        Unpin
                      </Button>
                    </motion.div>
                  </div>
                ))
              ) : (
                <div className="text-zinc-500 text-center py-4">No pinned facts</div>
              )}
            </div>
          </div>
        </motion.div>

        {/* Recent Events */}
        <motion.div 
          whileHover={{ y: -2, scale: 1.01 }} 
          whileTap={{ scale: 0.997 }} 
          transition={{ type: "spring", stiffness: 260, damping: 20 }}
        >
          <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] hover:bg-white/10 transition-colors p-6">
            <h3 className="text-xl font-semibold text-zinc-200 mb-4">🕐 Recent Events</h3>
            
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {recentEvents.length > 0 ? (
                recentEvents.map((event) => (
                  <div key={event.id} className="p-3 bg-white/5 border border-white/10 rounded">
                    <div className="flex items-center justify-between mb-2">
                      <span className={`px-2 py-1 text-xs rounded ${getEventTypeColor(event.type)} text-white`}>
                        {event.type}
                      </span>
                      <div className="text-xs text-zinc-500">{formatTimestamp(event.timestamp)}</div>
                    </div>
                    <div className="text-sm text-zinc-200">
                      <pre className="whitespace-pre-wrap">{JSON.stringify(event.payload, null, 2)}</pre>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-zinc-500 text-center py-4">No recent events</div>
              )}
            </div>
          </div>
        </motion.div>
      </div>

      {/* Context State */}
      {contextState && (
        <motion.div 
          whileHover={{ y: -2, scale: 1.01 }} 
          whileTap={{ scale: 0.997 }} 
          transition={{ type: "spring", stiffness: 260, damping: 20 }}
        >
          <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.35)] hover:bg-white/10 transition-colors p-6">
            <h3 className="text-xl font-semibold text-zinc-200 mb-4">🧠 Current Context State</h3>
            
            <div className="bg-white/5 border border-white/10 rounded p-4 overflow-x-auto">
              <pre className="text-zinc-200 text-sm">
                {JSON.stringify(contextState, null, 2)}
              </pre>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}