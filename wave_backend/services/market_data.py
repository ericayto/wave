"""
Market Data Service with CCXT Integration
Real-time price feeds, OHLCV data, and market data caching.
"""

import asyncio
import ccxt.async_support as ccxt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
from decimal import Decimal

from ..config.settings import get_settings
from ..services.event_bus import EventBus
from ..models.database import get_db

logger = logging.getLogger(__name__)

@dataclass
class Ticker:
    """Market ticker data."""
    symbol: str
    bid: float
    ask: float
    last: float
    high: float
    low: float
    volume: float
    change: float
    change_percent: float
    timestamp: datetime

@dataclass
class OHLCV:
    """OHLCV candlestick data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class OrderBookLevel:
    """Order book level."""
    price: float
    amount: float

@dataclass
class OrderBook:
    """Order book snapshot."""
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime

class MarketDataService:
    """Market data service with CCXT integration."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.settings = get_settings()
        
        # Exchange connections
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.active_symbols: set = set()
        
        # Data cache
        self.tickers: Dict[str, Ticker] = {}
        self.ohlcv_cache: Dict[Tuple[str, str], List[OHLCV]] = {}  # (symbol, timeframe) -> data
        self.order_books: Dict[str, OrderBook] = {}
        
        # Background tasks
        self.price_feed_task: Optional[asyncio.Task] = None
        self.data_update_task: Optional[asyncio.Task] = None
        
        # Rate limiting
        self.last_request_times: Dict[str, datetime] = {}
        self.rate_limit_delay = 1.0  # Seconds between requests
        
    async def start(self):
        """Start market data service."""
        logger.info("Starting market data service...")
        
        # Initialize exchanges
        await self._initialize_exchanges()
        
        # Start background tasks
        self.price_feed_task = asyncio.create_task(self._price_feed_loop())
        self.data_update_task = asyncio.create_task(self._data_update_loop())
        
        logger.info("Market data service started")
    
    async def stop(self):
        """Stop market data service."""
        logger.info("Stopping market data service...")
        
        # Cancel background tasks
        if self.price_feed_task:
            self.price_feed_task.cancel()
        if self.data_update_task:
            self.data_update_task.cancel()
        
        # Close exchange connections
        for exchange in self.exchanges.values():
            await exchange.close()
        
        logger.info("Market data service stopped")
    
    async def _initialize_exchanges(self):
        """Initialize exchange connections."""
        try:
            # Initialize Kraken
            if 'kraken' in self.settings.exchanges:
                kraken_config = self.settings.exchanges['kraken']
                
                self.exchanges['kraken'] = ccxt.kraken({
                    'apiKey': kraken_config.api_key,
                    'secret': kraken_config.api_secret,
                    'sandbox': kraken_config.sandbox,
                    'rateLimit': 1000,  # Kraken rate limit
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot'  # Spot trading only
                    }
                })
                
                # Load markets
                await self.exchanges['kraken'].load_markets()
                logger.info(f"Kraken initialized with {len(self.exchanges['kraken'].markets)} markets")
        
        except Exception as e:
            logger.error(f"Failed to initialize exchanges: {e}")
            # Continue with mock data if exchange initialization fails
            logger.info("Continuing with mock market data")
    
    async def subscribe_to_symbol(self, symbol: str):
        """Subscribe to real-time updates for a symbol."""
        self.active_symbols.add(symbol)
        logger.info(f"Subscribed to {symbol}")
    
    async def unsubscribe_from_symbol(self, symbol: str):
        """Unsubscribe from a symbol."""
        self.active_symbols.discard(symbol)
        logger.info(f"Unsubscribed from {symbol}")
    
    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """Get current ticker for a symbol."""
        if symbol in self.tickers:
            return self.tickers[symbol]
        
        # Fetch from exchange if not cached
        return await self._fetch_ticker(symbol)
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[OrderBook]:
        """Get order book for a symbol."""
        try:
            if 'kraken' in self.exchanges:
                exchange = self.exchanges['kraken']
                
                # Rate limiting
                await self._rate_limit_check('kraken')
                
                # Fetch order book
                order_book = await exchange.fetch_order_book(symbol, limit)
                
                bids = [OrderBookLevel(price=float(bid[0]), amount=float(bid[1])) 
                       for bid in order_book['bids'][:limit]]
                asks = [OrderBookLevel(price=float(ask[0]), amount=float(ask[1])) 
                       for ask in order_book['asks'][:limit]]
                
                result = OrderBook(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    timestamp=datetime.utcnow()
                )
                
                self.order_books[symbol] = result
                return result
                
        except Exception as e:
            logger.error(f"Failed to fetch order book for {symbol}: {e}")
            
        return self.order_books.get(symbol)
    
    async def get_ohlcv(self, 
                       symbol: str, 
                       timeframe: str = '1h',
                       limit: int = 100,
                       since: Optional[datetime] = None) -> List[OHLCV]:
        """Get OHLCV data for a symbol."""
        cache_key = (symbol, timeframe)
        
        # Check cache first
        if cache_key in self.ohlcv_cache:
            cached_data = self.ohlcv_cache[cache_key]
            if cached_data and len(cached_data) >= limit:
                return cached_data[-limit:]
        
        # Fetch from exchange
        return await self._fetch_ohlcv(symbol, timeframe, limit, since)
    
    async def _fetch_ticker(self, symbol: str) -> Optional[Ticker]:
        """Fetch ticker from exchange."""
        try:
            if 'kraken' in self.exchanges:
                exchange = self.exchanges['kraken']
                
                # Rate limiting
                await self._rate_limit_check('kraken')
                
                # Fetch ticker
                ticker_data = await exchange.fetch_ticker(symbol)
                
                ticker = Ticker(
                    symbol=symbol,
                    bid=float(ticker_data['bid']) if ticker_data['bid'] else 0.0,
                    ask=float(ticker_data['ask']) if ticker_data['ask'] else 0.0,
                    last=float(ticker_data['last']) if ticker_data['last'] else 0.0,
                    high=float(ticker_data['high']) if ticker_data['high'] else 0.0,
                    low=float(ticker_data['low']) if ticker_data['low'] else 0.0,
                    volume=float(ticker_data['baseVolume']) if ticker_data['baseVolume'] else 0.0,
                    change=float(ticker_data['change']) if ticker_data['change'] else 0.0,
                    change_percent=float(ticker_data['percentage']) if ticker_data['percentage'] else 0.0,
                    timestamp=datetime.utcnow()
                )
                
                self.tickers[symbol] = ticker
                return ticker
                
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
        
        # Return mock data if fetch fails
        return self._generate_mock_ticker(symbol)
    
    async def _fetch_ohlcv(self, 
                          symbol: str, 
                          timeframe: str, 
                          limit: int,
                          since: Optional[datetime] = None) -> List[OHLCV]:
        """Fetch OHLCV data from exchange."""
        try:
            if 'kraken' in self.exchanges:
                exchange = self.exchanges['kraken']
                
                # Rate limiting
                await self._rate_limit_check('kraken')
                
                # Convert since to timestamp
                since_timestamp = None
                if since:
                    since_timestamp = int(since.timestamp() * 1000)
                
                # Fetch OHLCV
                ohlcv_data = await exchange.fetch_ohlcv(
                    symbol, timeframe, since_timestamp, limit
                )
                
                ohlcv_list = []
                for candle in ohlcv_data:
                    ohlcv_list.append(OHLCV(
                        timestamp=datetime.fromtimestamp(candle[0] / 1000),
                        open=float(candle[1]),
                        high=float(candle[2]),
                        low=float(candle[3]),
                        close=float(candle[4]),
                        volume=float(candle[5])
                    ))
                
                # Cache the data
                cache_key = (symbol, timeframe)
                self.ohlcv_cache[cache_key] = ohlcv_list
                
                return ohlcv_list
                
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
        
        # Return mock data if fetch fails
        return self._generate_mock_ohlcv(symbol, timeframe, limit)
    
    async def _rate_limit_check(self, exchange_name: str):
        """Check rate limits before making requests."""
        now = datetime.utcnow()
        last_request = self.last_request_times.get(exchange_name, datetime.min)
        
        time_since_last = (now - last_request).total_seconds()
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_times[exchange_name] = datetime.utcnow()
    
    async def _price_feed_loop(self):
        """Background task for real-time price updates."""
        while True:
            try:
                if self.active_symbols:
                    # Update tickers for active symbols
                    for symbol in list(self.active_symbols):
                        try:
                            ticker = await self._fetch_ticker(symbol)
                            if ticker:
                                # Emit ticker update event
                                await self.event_bus.publish("market_data", {
                                    "type": "ticker_update",
                                    "symbol": symbol,
                                    "data": {
                                        "price": ticker.last,
                                        "bid": ticker.bid,
                                        "ask": ticker.ask,
                                        "change_24h": ticker.change,
                                        "change_24h_pct": ticker.change_percent,
                                        "volume": ticker.volume,
                                        "timestamp": ticker.timestamp.isoformat()
                                    }
                                })
                        except Exception as e:
                            logger.error(f"Error updating ticker for {symbol}: {e}")
                
                # Wait before next update (respecting rate limits)
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in price feed loop: {e}")
                await asyncio.sleep(10)
    
    async def _data_update_loop(self):
        """Background task for updating OHLCV data."""
        while True:
            try:
                if self.active_symbols:
                    # Update OHLCV data for active symbols
                    timeframes = ['1m', '5m', '1h', '4h', '1d']
                    
                    for symbol in list(self.active_symbols):
                        for timeframe in timeframes:
                            try:
                                # Get latest 100 candles
                                ohlcv_data = await self._fetch_ohlcv(symbol, timeframe, 100)
                                
                                if ohlcv_data:
                                    # Emit OHLCV update event
                                    await self.event_bus.publish("market_data", {
                                        "type": "ohlcv_update",
                                        "symbol": symbol,
                                        "timeframe": timeframe,
                                        "data": [
                                            {
                                                "timestamp": candle.timestamp.isoformat(),
                                                "open": candle.open,
                                                "high": candle.high,
                                                "low": candle.low,
                                                "close": candle.close,
                                                "volume": candle.volume
                                            }
                                            for candle in ohlcv_data[-50:]  # Last 50 candles
                                        ]
                                    })
                            
                            except Exception as e:
                                logger.error(f"Error updating OHLCV for {symbol} {timeframe}: {e}")
                            
                            # Small delay between timeframes
                            await asyncio.sleep(2)
                
                # Update every 5 minutes
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data update loop: {e}")
                await asyncio.sleep(30)
    
    def _generate_mock_ticker(self, symbol: str) -> Ticker:
        """Generate mock ticker data for testing."""
        import random
        
        # Base prices for different symbols
        base_prices = {
            'BTC/USDT': 45000.0,
            'ETH/USDT': 2800.0,
            'ADA/USDT': 0.5,
            'SOL/USDT': 95.0,
            'MATIC/USDT': 0.8,
            'AVAX/USDT': 35.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Add random variation
        variation = random.uniform(0.98, 1.02)
        current_price = base_price * variation
        
        # Create spread
        spread_pct = random.uniform(0.0001, 0.001)
        spread = current_price * spread_pct
        
        # Random change
        change_pct = random.uniform(-5.0, 5.0)
        change = current_price * (change_pct / 100)
        
        return Ticker(
            symbol=symbol,
            bid=current_price - spread/2,
            ask=current_price + spread/2,
            last=current_price,
            high=current_price * 1.05,
            low=current_price * 0.95,
            volume=random.uniform(100000, 10000000),
            change=change,
            change_percent=change_pct,
            timestamp=datetime.utcnow()
        )
    
    def _generate_mock_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[OHLCV]:
        """Generate mock OHLCV data for testing."""
        import random
        
        # Get timeframe in minutes
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '8h': 480,
            '12h': 720, '1d': 1440, '3d': 4320, '1w': 10080
        }
        
        interval_minutes = timeframe_minutes.get(timeframe, 60)
        
        # Base price
        base_prices = {
            'BTC/USDT': 45000.0,
            'ETH/USDT': 2800.0,
            'ADA/USDT': 0.5,
            'SOL/USDT': 95.0,
        }
        
        base_price = base_prices.get(symbol, 100.0)
        current_price = base_price
        
        ohlcv_data = []
        end_time = datetime.utcnow()
        
        for i in range(limit):
            timestamp = end_time - timedelta(minutes=interval_minutes * (limit - i - 1))
            
            # Generate realistic price movement
            price_change = random.uniform(-0.02, 0.02)  # ±2% per candle
            new_price = current_price * (1 + price_change)
            
            # Create OHLC
            open_price = current_price
            close_price = new_price
            high_price = max(open_price, close_price) * random.uniform(1.0, 1.01)
            low_price = min(open_price, close_price) * random.uniform(0.99, 1.0)
            volume = random.uniform(100, 10000)
            
            ohlcv_data.append(OHLCV(
                timestamp=timestamp,
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=round(volume, 4)
            ))
            
            current_price = close_price
        
        return ohlcv_data
    
    async def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols."""
        symbols = []
        
        try:
            if 'kraken' in self.exchanges:
                exchange = self.exchanges['kraken']
                markets = exchange.markets
                
                for symbol in markets:
                    if markets[symbol]['active'] and markets[symbol]['type'] == 'spot':
                        symbols.append(symbol)
        
        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
        
        # Return default symbols if exchange unavailable
        if not symbols:
            symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'MATIC/USDT', 'AVAX/USDT']
        
        return sorted(symbols)
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """Get market overview summary."""
        symbols = await self.get_available_symbols()
        active_symbols = len(self.active_symbols)
        
        # Get some sample tickers
        sample_tickers = {}
        for symbol in symbols[:5]:  # First 5 symbols
            ticker = await self.get_ticker(symbol)
            if ticker:
                sample_tickers[symbol] = {
                    'price': ticker.last,
                    'change_24h_pct': ticker.change_percent,
                    'volume': ticker.volume
                }
        
        return {
            'total_symbols': len(symbols),
            'active_subscriptions': active_symbols,
            'sample_tickers': sample_tickers,
            'last_updated': datetime.utcnow().isoformat()
        }