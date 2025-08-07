// API Types and Interfaces

export interface ApiResponse<T = any> {
  success?: boolean;
  data?: T;
  message?: string;
  error?: string;
}

// Portfolio Types
export interface Balance {
  symbol: string;
  free: number;
  used: number;
  total: number;
}

export interface Position {
  symbol: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  market_value: number;
}

export interface PortfolioSummary {
  total_value: number;
  cash_balance: number;
  invested_value: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  total_pnl: number;
  total_pnl_pct: number;
}

export interface PortfolioData {
  summary: PortfolioSummary;
  balances: Balance[];
  positions: Position[];
  last_updated: string;
}

// Market Types
export interface MarketTicker {
  symbol: string;
  price: number;
  change_24h: number;
  change_24h_pct: number;
  volume_24h: number;
  high_24h: number;
  low_24h: number;
  last_updated: string;
}

export interface OHLCV {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// Trading Types
export type OrderSide = 'buy' | 'sell';
export type OrderType = 'market' | 'limit';
export type OrderStatus = 'pending' | 'open' | 'filled' | 'canceled' | 'rejected';

export interface Order {
  id: string;
  symbol: string;
  side: OrderSide;
  quantity: number;
  filled_quantity: number;
  type: OrderType;
  price?: number;
  avg_fill_price?: number;
  status: OrderStatus;
  created_at: string;
  updated_at: string;
}

export interface Fill {
  id: string;
  order_id: string;
  symbol: string;
  side: OrderSide;
  quantity: number;
  price: number;
  fee: number;
  timestamp: string;
}

// Strategy Types
export type StrategyStatus = 'active' | 'inactive' | 'paused' | 'error';

export interface Strategy {
  id: string;
  name: string;
  version: string;
  status: StrategyStatus;
  description: string;
  parameters: Record<string, any>;
  created_at: string;
  updated_at: string;
}

// Risk Types
export interface RiskLimits {
  max_position_pct: number;
  daily_loss_limit_pct: number;
  max_orders_per_hour: number;
  circuit_breaker_spread_bps: number;
  updated_at: string;
}

export interface RiskStatus {
  current_exposure_pct: number;
  daily_loss_pct: number;
  orders_last_hour: number;
  circuit_breaker_triggered: boolean;
  risk_score: number;
  status: 'healthy' | 'warning' | 'critical';
}

// Log Types
export type LogLevel = 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
export type LogType = 'decisions' | 'orders' | 'audit' | 'strategy' | 'risk';

export interface LogEntry {
  id: string;
  timestamp: string;
  level: LogLevel;
  type: LogType;
  message: string;
  details: Record<string, any>;
}

// Memory Types
export interface StateProjection {
  keys: string[];
  data: Record<string, any>;
  timestamp: string;
}

export interface Event {
  id: number;
  timestamp: string;
  type: string;
  payload: Record<string, any>;
}

export interface MemorySnippet {
  id: number;
  relevance_score: number;
  content: string;
  metadata: Record<string, any>;
  timestamp: string;
}

export interface PinnedFact {
  key: string;
  value: string;
  ttl?: string;
  created_at: string;
  updated_at: string;
}

export interface MemoryStats {
  total_events: number;
  total_summaries: number;
  total_pinned_facts: number;
  context_usage_pct: number;
  last_summary: string;
  next_summary: string;
}

// WebSocket Types
export interface WebSocketMessage {
  topic: string;
  type: string;
  data: any;
  timestamp: string;
}

export interface WebSocketSubscription {
  type: 'subscribe' | 'unsubscribe';
  topics: string[];
}

// UI State Types
export interface AppState {
  isConnected: boolean;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  lastUpdate: string;
  theme: 'ocean' | 'dark' | 'light';
}

export interface DashboardMetrics {
  portfolio_value: number;
  daily_pnl: number;
  active_strategies: number;
  open_orders: number;
  risk_score: number;
  bot_status: 'running' | 'paused' | 'stopped' | 'error';
}

// Chart Types
export interface ChartDataPoint {
  timestamp: string;
  value: number;
  label?: string;
}

export interface PnLDataPoint {
  date: string;
  realized_pnl: number;
  unrealized_pnl: number;
  total_pnl: number;
  portfolio_value: number;
}

// Component Props Types
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export interface LoadingProps {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
}

export interface EmptyStateProps {
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}