import { useState } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { WebSocketProvider } from './hooks/useWebSocket'
import { Layout } from './components/Layout'
import { Dashboard } from './pages/Dashboard'
import { Portfolio } from './pages/Portfolio'
import { Strategies } from './pages/Strategies'
import { Trading } from './pages/Trading'
import { Settings } from './pages/Settings'
import LLMCenter from './pages/LLMCenter'
import StrategyGenerator from './pages/StrategyGenerator'
import MemoryInspector from './pages/MemoryInspector'

// Create a query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30 * 1000, // 30 seconds
      gcTime: 5 * 60 * 1000, // 5 minutes
      retry: (failureCount, error) => {
        // Don't retry on 404s or client errors
        if (error instanceof Error && error.message.includes('404')) {
          return false
        }
        return failureCount < 3
      },
    },
  },
})

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard')

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />
      case 'portfolio':
        return <Portfolio />
      case 'strategies':
        return <Strategies />
      case 'trading':
        return <Trading />
      case 'llm-center':
        return <LLMCenter />
      case 'strategy-generator':
        return <StrategyGenerator />
      case 'memory-inspector':
        return <MemoryInspector />
      case 'settings':
        return <Settings />
      default:
        return <Dashboard />
    }
  }

  return (
    <QueryClientProvider client={queryClient}>
      <WebSocketProvider url="ws://localhost:8080/ws/stream">
        <div className="min-h-screen text-fg-primary">
          <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
            {renderPage()}
          </Layout>
        </div>
      </WebSocketProvider>
    </QueryClientProvider>
  )
}

export default App