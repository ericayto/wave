import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react'
import { WebSocketMessage, WebSocketSubscription } from '../types'

interface WebSocketContextType {
  isConnected: boolean
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error'
  subscribe: (topics: string[]) => void
  unsubscribe: (topics: string[]) => void
  lastMessage: WebSocketMessage | null
  messages: WebSocketMessage[]
  sendMessage: (message: any) => void
}

const WebSocketContext = createContext<WebSocketContextType | null>(null)

interface WebSocketProviderProps {
  url: string
  children: React.ReactNode
  reconnectInterval?: number
  maxReconnectAttempts?: number
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({
  url,
  children,
  reconnectInterval = 3000,
  maxReconnectAttempts = 5,
}) => {
  const ws = useRef<WebSocket | null>(null)
  const reconnectTimeoutId = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttempts = useRef(0)
  
  const [isConnected, setIsConnected] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected')
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [messages, setMessages] = useState<WebSocketMessage[]>([])
  
  const connect = useCallback(() => {
    try {
      setConnectionStatus('connecting')
      ws.current = new WebSocket(url)
      
      ws.current.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        setConnectionStatus('connected')
        reconnectAttempts.current = 0
      }
      
      ws.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          setLastMessage(message)
          setMessages(prev => [...prev.slice(-99), message]) // Keep last 100 messages
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }
      
      ws.current.onclose = () => {
        console.log('WebSocket disconnected')
        setIsConnected(false)
        setConnectionStatus('disconnected')
        
        // Attempt reconnection
        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current += 1
          console.log(`Attempting reconnection ${reconnectAttempts.current}/${maxReconnectAttempts}`)
          
          reconnectTimeoutId.current = setTimeout(() => {
            connect()
          }, reconnectInterval)
        } else {
          console.log('Max reconnection attempts reached')
          setConnectionStatus('error')
        }
      }
      
      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        setConnectionStatus('error')
      }
      
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      setConnectionStatus('error')
    }
  }, [url, reconnectInterval, maxReconnectAttempts])
  
  useEffect(() => {
    connect()
    
    return () => {
      if (reconnectTimeoutId.current) {
        clearTimeout(reconnectTimeoutId.current)
      }
      if (ws.current) {
        ws.current.close()
      }
    }
  }, [connect])
  
  const sendMessage = useCallback((message: any) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket is not connected')
    }
  }, [])
  
  const subscribe = useCallback((topics: string[]) => {
    const message: WebSocketSubscription = {
      type: 'subscribe',
      topics,
    }
    sendMessage(message)
  }, [sendMessage])
  
  const unsubscribe = useCallback((topics: string[]) => {
    const message: WebSocketSubscription = {
      type: 'unsubscribe',
      topics,
    }
    sendMessage(message)
  }, [sendMessage])
  
  const value: WebSocketContextType = {
    isConnected,
    connectionStatus,
    subscribe,
    unsubscribe,
    lastMessage,
    messages,
    sendMessage,
  }
  
  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  )
}

export const useWebSocket = () => {
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider')
  }
  return context
}