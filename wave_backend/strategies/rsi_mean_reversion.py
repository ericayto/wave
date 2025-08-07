"""
RSI Mean Reversion Strategy
RSI-based mean reversion strategy for short-term trading.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from ..services.strategy_engine import BaseStrategy, TradingSignal, SignalType
from ..services.market_data import OHLCV

logger = logging.getLogger(__name__)

class RSIMeanReversionStrategy(BaseStrategy):
    """RSI Mean Reversion Strategy."""
    
    def __init__(self, strategy_id: str, config: Dict[str, Any]):
        super().__init__(strategy_id, config)
        
        # Strategy parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.oversold_level = config.get('oversold_level', 30)
        self.overbought_level = config.get('overbought_level', 70)
        self.extreme_oversold = config.get('extreme_oversold', 20)  # Extreme levels
        self.extreme_overbought = config.get('extreme_overbought', 80)
        self.exit_rsi_level = config.get('exit_rsi_level', 50)  # Exit near midline
        
        # Risk management
        self.stop_loss_pct = config.get('stop_loss_pct', 0.03)  # 3% stop loss
        self.take_profit_pct = config.get('take_profit_pct', 0.02)  # 2% take profit
        self.max_hold_hours = config.get('max_hold_hours', 24)  # Max holding period
        
        # Filters
        self.use_volatility_filter = config.get('use_volatility_filter', True)
        self.max_volatility_pct = config.get('max_volatility_pct', 0.04)  # 4% max volatility
        self.use_volume_confirmation = config.get('use_volume_confirmation', True)
        self.volume_threshold = config.get('volume_threshold', 1.5)  # 50% above average
        
        # Trend filter
        self.use_trend_filter = config.get('use_trend_filter', True)
        self.trend_sma_period = config.get('trend_sma_period', 200)
        
        # State tracking
        self.positions: Dict[str, Dict] = {}  # Track open positions per symbol
        self.last_signals: Dict[str, Dict] = {}  # Track last signal per symbol
    
    def get_required_indicators(self) -> List[Dict[str, Any]]:
        """Get required indicators for this strategy."""
        indicators = [
            {'name': 'RSI', 'params': {'period': self.rsi_period}},
            {'name': 'ATR', 'params': {'period': 14}},  # For volatility
            {'name': 'SMA', 'params': {'period': 20}},  # Short-term trend
        ]
        
        if self.use_trend_filter:
            indicators.append({'name': 'SMA', 'params': {'period': self.trend_sma_period}})
        
        return indicators
    
    async def analyze(self, 
                     symbol: str, 
                     market_data: List[OHLCV], 
                     indicators: Dict[str, Any]) -> Optional[TradingSignal]:
        """Analyze market data and generate trading signals."""
        
        try:
            if len(market_data) < max(self.rsi_period, self.trend_sma_period) + 10:
                logger.debug(f"Insufficient data for {symbol}")
                return None
            
            # Get current market data
            current_candle = market_data[-1]
            previous_candle = market_data[-2] if len(market_data) > 1 else current_candle
            current_price = current_candle.close
            current_volume = current_candle.volume
            
            # Get RSI values
            if 'RSI' not in indicators or not hasattr(indicators['RSI'], 'values'):
                logger.debug(f"No RSI data for {symbol}")
                return None
            
            rsi_values = indicators['RSI'].values
            if len(rsi_values) < 2:
                return None
                
            current_rsi = rsi_values[-1]
            previous_rsi = rsi_values[-2]
            
            # Get ATR for volatility assessment
            atr_value = 0.0
            if 'ATR' in indicators and hasattr(indicators['ATR'], 'values'):
                atr_values = indicators['ATR'].values
                if atr_values and len(atr_values) > 0:
                    atr_value = atr_values[-1]
            
            # Calculate volatility
            volatility_pct = atr_value / current_price if current_price > 0 else 0
            
            # Volume analysis
            volumes = [candle.volume for candle in market_data[-20:]]
            avg_volume = sum(volumes) / len(volumes) if volumes else 1
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Trend analysis
            trend_direction = 'neutral'
            if self.use_trend_filter and 'SMA' in indicators:
                # This would need to be handled better with multiple SMAs
                # For now, use simple price vs SMA
                closes = [candle.close for candle in market_data]
                trend_sma = sum(closes[-self.trend_sma_period:]) / self.trend_sma_period
                
                if current_price > trend_sma * 1.01:  # 1% above
                    trend_direction = 'up'
                elif current_price < trend_sma * 0.99:  # 1% below
                    trend_direction = 'down'
            
            # Apply filters
            volatility_ok = True
            if self.use_volatility_filter:
                volatility_ok = volatility_pct <= self.max_volatility_pct
            
            volume_ok = True
            if self.use_volume_confirmation:
                volume_ok = volume_ratio >= self.volume_threshold
            
            # Check existing position
            existing_position = self.positions.get(symbol)
            
            if existing_position:
                # Manage existing position
                return await self._manage_existing_position(
                    symbol, current_price, current_rsi, existing_position, market_data
                )
            
            # Look for new entry signals
            signal = None
            
            # Oversold bounce (long entry)
            if (current_rsi <= self.oversold_level and 
                previous_rsi > current_rsi and  # RSI is declining into oversold
                volatility_ok and 
                (not self.use_volume_confirmation or volume_ok) and
                (not self.use_trend_filter or trend_direction != 'down')):
                
                # Calculate signal strength based on how oversold
                if current_rsi <= self.extreme_oversold:
                    strength = 0.9  # Very strong signal
                else:
                    strength = 0.6 + (self.oversold_level - current_rsi) / self.oversold_level * 0.3
                
                # Calculate stop loss and take profit
                stop_loss_price = current_price * (1 - self.stop_loss_pct)
                take_profit_price = current_price * (1 + self.take_profit_pct)
                
                # Track position
                self.positions[symbol] = {
                    'side': 'long',
                    'entry_price': current_price,
                    'entry_time': datetime.utcnow(),
                    'entry_rsi': current_rsi,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price
                }
                
                signal = TradingSignal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                    metadata={
                        'rsi': current_rsi,
                        'previous_rsi': previous_rsi,
                        'oversold_level': self.oversold_level,
                        'volatility_pct': volatility_pct,
                        'volume_ratio': volume_ratio,
                        'trend_direction': trend_direction,
                        'entry_reason': 'RSI oversold mean reversion',
                        'signal_strength_factors': {
                            'extreme_oversold': current_rsi <= self.extreme_oversold,
                            'rsi_declining': previous_rsi > current_rsi,
                            'volatility_ok': volatility_ok,
                            'volume_ok': volume_ok
                        }
                    }
                )
            
            # Overbought fade (short entry) - only if trend allows
            elif (current_rsi >= self.overbought_level and 
                  previous_rsi < current_rsi and  # RSI is rising into overbought
                  volatility_ok and
                  (not self.use_volume_confirmation or volume_ok) and
                  (not self.use_trend_filter or trend_direction != 'up')):
                
                # Calculate signal strength based on how overbought
                if current_rsi >= self.extreme_overbought:
                    strength = 0.9  # Very strong signal
                else:
                    strength = 0.6 + (current_rsi - self.overbought_level) / (100 - self.overbought_level) * 0.3
                
                # Calculate stop loss and take profit
                stop_loss_price = current_price * (1 + self.stop_loss_pct)  # Above for short
                take_profit_price = current_price * (1 - self.take_profit_pct)  # Below for short
                
                # Track position
                self.positions[symbol] = {
                    'side': 'short',
                    'entry_price': current_price,
                    'entry_time': datetime.utcnow(),
                    'entry_rsi': current_rsi,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price
                }
                
                signal = TradingSignal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=current_price,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                    metadata={
                        'rsi': current_rsi,
                        'previous_rsi': previous_rsi,
                        'overbought_level': self.overbought_level,
                        'volatility_pct': volatility_pct,
                        'volume_ratio': volume_ratio,
                        'trend_direction': trend_direction,
                        'entry_reason': 'RSI overbought mean reversion',
                        'signal_strength_factors': {
                            'extreme_overbought': current_rsi >= self.extreme_overbought,
                            'rsi_rising': previous_rsi < current_rsi,
                            'volatility_ok': volatility_ok,
                            'volume_ok': volume_ok
                        }
                    }
                )
            
            # Record last signal attempt
            self.last_signals[symbol] = {
                'timestamp': datetime.utcnow(),
                'rsi': current_rsi,
                'price': current_price,
                'signal_generated': signal is not None,
                'filters': {
                    'volatility_ok': volatility_ok,
                    'volume_ok': volume_ok,
                    'trend_direction': trend_direction
                }
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in RSI mean reversion analysis for {symbol}: {e}")
            return None
    
    async def _manage_existing_position(self, 
                                      symbol: str, 
                                      current_price: float, 
                                      current_rsi: float,
                                      position: Dict,
                                      market_data: List[OHLCV]) -> Optional[TradingSignal]:
        """Manage existing position."""
        
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        side = position['side']
        
        # Calculate current P&L
        if side == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # short
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Check time-based exit
        time_in_position = datetime.utcnow() - entry_time
        time_exit = time_in_position > timedelta(hours=self.max_hold_hours)
        
        # Check RSI-based exit (mean reversion completion)
        rsi_exit = False
        if side == 'long':
            # Exit long when RSI returns to neutral/overbought
            rsi_exit = current_rsi >= self.exit_rsi_level
        else:  # short
            # Exit short when RSI returns to neutral/oversold
            rsi_exit = current_rsi <= self.exit_rsi_level
        
        # Check stop loss
        stop_loss_hit = False
        if side == 'long':
            stop_loss_hit = current_price <= position['stop_loss']
        else:  # short
            stop_loss_hit = current_price >= position['stop_loss']
        
        # Check take profit
        take_profit_hit = False
        if side == 'long':
            take_profit_hit = current_price >= position['take_profit']
        else:  # short
            take_profit_hit = current_price <= position['take_profit']
        
        # Determine exit reason and strength
        exit_signal = None
        if stop_loss_hit:
            exit_signal = ('stop_loss', 1.0)
        elif take_profit_hit:
            exit_signal = ('take_profit', 0.9)
        elif time_exit:
            exit_signal = ('time_limit', 0.7)
        elif rsi_exit:
            exit_signal = ('rsi_mean_reversion', 0.8)
        
        if exit_signal:
            exit_reason, strength = exit_signal
            
            # Remove position
            del self.positions[symbol]
            
            # Determine signal type for exit
            if side == 'long':
                signal_type = SignalType.SELL
            else:  # short position, need to buy to close
                signal_type = SignalType.BUY
            
            return TradingSignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                price=current_price,
                metadata={
                    'exit_reason': exit_reason,
                    'position_side': side,
                    'entry_price': entry_price,
                    'entry_rsi': position['entry_rsi'],
                    'exit_rsi': current_rsi,
                    'pnl_pct': pnl_pct,
                    'time_in_position_hours': time_in_position.total_seconds() / 3600,
                    'exit_conditions': {
                        'stop_loss_hit': stop_loss_hit,
                        'take_profit_hit': take_profit_hit,
                        'time_exit': time_exit,
                        'rsi_exit': rsi_exit
                    }
                }
            )
        
        return None
    
    def validate_config(self) -> List[str]:
        """Validate strategy configuration."""
        errors = super().validate_config()
        
        if not (5 <= self.rsi_period <= 50):
            errors.append("RSI period must be between 5 and 50")
        
        if not (10 <= self.oversold_level <= 40):
            errors.append("Oversold level must be between 10 and 40")
        
        if not (60 <= self.overbought_level <= 90):
            errors.append("Overbought level must be between 60 and 90")
        
        if self.oversold_level >= self.overbought_level:
            errors.append("Oversold level must be less than overbought level")
        
        if not (5 <= self.extreme_oversold <= self.oversold_level):
            errors.append("Extreme oversold must be between 5 and oversold level")
        
        if not (self.overbought_level <= self.extreme_overbought <= 95):
            errors.append("Extreme overbought must be between overbought level and 95")
        
        if not (0.01 <= self.stop_loss_pct <= 0.1):
            errors.append("Stop loss must be between 1% and 10%")
        
        if not (0.005 <= self.take_profit_pct <= 0.05):
            errors.append("Take profit must be between 0.5% and 5%")
        
        if self.max_hold_hours < 1:
            errors.append("Max hold hours must be at least 1")
        
        return errors
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and current state."""
        base_info = super().get_status()
        
        return {
            **base_info,
            'strategy_type': 'RSI Mean Reversion',
            'description': f'RSI mean reversion strategy (RSI {self.rsi_period}, {self.oversold_level}/{self.overbought_level})',
            'parameters': {
                'rsi_period': self.rsi_period,
                'oversold_level': self.oversold_level,
                'overbought_level': self.overbought_level,
                'extreme_oversold': self.extreme_oversold,
                'extreme_overbought': self.extreme_overbought,
                'exit_rsi_level': self.exit_rsi_level,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'max_hold_hours': self.max_hold_hours,
                'use_volatility_filter': self.use_volatility_filter,
                'max_volatility_pct': self.max_volatility_pct,
                'use_volume_confirmation': self.use_volume_confirmation,
                'volume_threshold': self.volume_threshold,
                'use_trend_filter': self.use_trend_filter
            },
            'current_positions': {
                symbol: {
                    **pos,
                    'entry_time': pos['entry_time'].isoformat(),
                    'time_in_position_hours': (datetime.utcnow() - pos['entry_time']).total_seconds() / 3600
                }
                for symbol, pos in self.positions.items()
            },
            'last_signals': {
                symbol: {
                    **signal,
                    'timestamp': signal['timestamp'].isoformat()
                }
                for symbol, signal in self.last_signals.items()
            },
            'active_positions': len(self.positions)
        }

def create_rsi_mean_reversion_strategy(strategy_id: str, config: Dict[str, Any]) -> RSIMeanReversionStrategy:
    """Factory function to create RSI Mean Reversion strategy."""
    
    # Default configuration
    default_config = {
        'name': 'RSI Mean Reversion',
        'symbols': ['ETH/USDT'],
        'timeframe': '15m',
        'rsi_period': 14,
        'oversold_level': 30,
        'overbought_level': 70,
        'extreme_oversold': 20,
        'extreme_overbought': 80,
        'exit_rsi_level': 50,
        'stop_loss_pct': 0.03,
        'take_profit_pct': 0.02,
        'max_hold_hours': 24,
        'use_volatility_filter': True,
        'max_volatility_pct': 0.04,
        'use_volume_confirmation': True,
        'volume_threshold': 1.5,
        'use_trend_filter': True,
        'trend_sma_period': 200
    }
    
    # Merge with provided config
    merged_config = {**default_config, **config}
    
    return RSIMeanReversionStrategy(strategy_id, merged_config)