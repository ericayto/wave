"""
Technical Indicators Library
Implementation of common trading indicators using pandas and numpy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class IndicatorResult:
    """Result of an indicator calculation."""
    name: str
    values: List[float]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]

class TechnicalIndicators:
    """Technical indicators calculator."""
    
    @staticmethod
    def sma(prices: List[float], period: int = 20) -> IndicatorResult:
        """Simple Moving Average."""
        df = pd.DataFrame({'price': prices})
        sma_values = df['price'].rolling(window=period).mean()
        
        return IndicatorResult(
            name='SMA',
            values=sma_values.fillna(0).tolist(),
            parameters={'period': period},
            metadata={'description': f'Simple Moving Average ({period} periods)'}
        )
    
    @staticmethod
    def ema(prices: List[float], period: int = 20) -> IndicatorResult:
        """Exponential Moving Average."""
        df = pd.DataFrame({'price': prices})
        ema_values = df['price'].ewm(span=period).mean()
        
        return IndicatorResult(
            name='EMA',
            values=ema_values.fillna(0).tolist(),
            parameters={'period': period},
            metadata={'description': f'Exponential Moving Average ({period} periods)'}
        )
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> IndicatorResult:
        """Relative Strength Index."""
        df = pd.DataFrame({'price': prices})
        delta = df['price'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi_values = 100 - (100 / (1 + rs))
        
        return IndicatorResult(
            name='RSI',
            values=rsi_values.fillna(50).tolist(),
            parameters={'period': period},
            metadata={
                'description': f'Relative Strength Index ({period} periods)',
                'overbought_level': 70,
                'oversold_level': 30
            }
        )
    
    @staticmethod
    def macd(prices: List[float], 
             fast_period: int = 12, 
             slow_period: int = 26, 
             signal_period: int = 9) -> IndicatorResult:
        """Moving Average Convergence Divergence."""
        df = pd.DataFrame({'price': prices})
        
        # Calculate MACD line
        ema_fast = df['price'].ewm(span=fast_period).mean()
        ema_slow = df['price'].ewm(span=slow_period).mean()
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return IndicatorResult(
            name='MACD',
            values=macd_line.fillna(0).tolist(),
            parameters={
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period
            },
            metadata={
                'description': f'MACD ({fast_period}, {slow_period}, {signal_period})',
                'signal_line': signal_line.fillna(0).tolist(),
                'histogram': histogram.fillna(0).tolist()
            }
        )
    
    @staticmethod
    def bollinger_bands(prices: List[float], 
                       period: int = 20, 
                       std_dev: float = 2.0) -> IndicatorResult:
        """Bollinger Bands."""
        df = pd.DataFrame({'price': prices})
        
        # Calculate middle band (SMA)
        middle_band = df['price'].rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = df['price'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        return IndicatorResult(
            name='BOLLINGER_BANDS',
            values=middle_band.fillna(0).tolist(),
            parameters={'period': period, 'std_dev': std_dev},
            metadata={
                'description': f'Bollinger Bands ({period}, {std_dev})',
                'upper_band': upper_band.fillna(0).tolist(),
                'lower_band': lower_band.fillna(0).tolist()
            }
        )
    
    @staticmethod
    def stochastic(highs: List[float], 
                   lows: List[float], 
                   closes: List[float],
                   k_period: int = 14,
                   d_period: int = 3) -> IndicatorResult:
        """Stochastic Oscillator."""
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        
        # Calculate %K
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (smooth %K)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return IndicatorResult(
            name='STOCHASTIC',
            values=k_percent.fillna(50).tolist(),
            parameters={'k_period': k_period, 'd_period': d_period},
            metadata={
                'description': f'Stochastic Oscillator ({k_period}, {d_period})',
                'd_values': d_percent.fillna(50).tolist(),
                'overbought_level': 80,
                'oversold_level': 20
            }
        )
    
    @staticmethod
    def atr(highs: List[float], 
            lows: List[float], 
            closes: List[float],
            period: int = 14) -> IndicatorResult:
        """Average True Range."""
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        
        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        atr_values = df['true_range'].rolling(window=period).mean()
        
        return IndicatorResult(
            name='ATR',
            values=atr_values.fillna(0).tolist(),
            parameters={'period': period},
            metadata={'description': f'Average True Range ({period} periods)'}
        )
    
    @staticmethod
    def williams_r(highs: List[float], 
                   lows: List[float], 
                   closes: List[float],
                   period: int = 14) -> IndicatorResult:
        """Williams %R."""
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        
        # Calculate highest high and lowest low
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        # Calculate Williams %R
        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        
        return IndicatorResult(
            name='WILLIAMS_R',
            values=williams_r.fillna(-50).tolist(),
            parameters={'period': period},
            metadata={
                'description': f'Williams %R ({period} periods)',
                'overbought_level': -20,
                'oversold_level': -80
            }
        )
    
    @staticmethod
    def cci(highs: List[float], 
            lows: List[float], 
            closes: List[float],
            period: int = 20) -> IndicatorResult:
        """Commodity Channel Index."""
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate SMA of typical price
        sma_tp = typical_price.rolling(window=period).mean()
        
        # Calculate mean deviation
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )
        
        # Calculate CCI
        cci_values = (typical_price - sma_tp) / (0.015 * mad)
        
        return IndicatorResult(
            name='CCI',
            values=cci_values.fillna(0).tolist(),
            parameters={'period': period},
            metadata={
                'description': f'Commodity Channel Index ({period} periods)',
                'overbought_level': 100,
                'oversold_level': -100
            }
        )
    
    @staticmethod
    def obv(closes: List[float], volumes: List[float]) -> IndicatorResult:
        """On Balance Volume."""
        df = pd.DataFrame({'close': closes, 'volume': volumes})
        
        # Calculate price direction
        df['price_direction'] = np.where(df['close'].diff() > 0, 1, 
                                        np.where(df['close'].diff() < 0, -1, 0))
        
        # Calculate OBV
        df['obv'] = (df['volume'] * df['price_direction']).cumsum()
        
        return IndicatorResult(
            name='OBV',
            values=df['obv'].fillna(0).tolist(),
            parameters={},
            metadata={'description': 'On Balance Volume'}
        )
    
    @staticmethod
    def adx(highs: List[float], 
            lows: List[float], 
            closes: List[float],
            period: int = 14) -> IndicatorResult:
        """Average Directional Index."""
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        
        # Calculate True Range and Directional Movement
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['prev_close'] = df['close'].shift(1)
        
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['prev_close']),
                abs(df['low'] - df['prev_close'])
            )
        )
        
        df['dm_plus'] = np.where(
            (df['high'] - df['prev_high']) > (df['prev_low'] - df['low']),
            np.maximum(df['high'] - df['prev_high'], 0),
            0
        )
        
        df['dm_minus'] = np.where(
            (df['prev_low'] - df['low']) > (df['high'] - df['prev_high']),
            np.maximum(df['prev_low'] - df['low'], 0),
            0
        )
        
        # Calculate smoothed averages
        df['atr'] = df['tr'].rolling(window=period).mean()
        df['di_plus'] = 100 * (df['dm_plus'].rolling(window=period).mean() / df['atr'])
        df['di_minus'] = 100 * (df['dm_minus'].rolling(window=period).mean() / df['atr'])
        
        # Calculate DX and ADX
        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        adx_values = df['dx'].rolling(window=period).mean()
        
        return IndicatorResult(
            name='ADX',
            values=adx_values.fillna(0).tolist(),
            parameters={'period': period},
            metadata={
                'description': f'Average Directional Index ({period} periods)',
                'di_plus': df['di_plus'].fillna(0).tolist(),
                'di_minus': df['di_minus'].fillna(0).tolist()
            }
        )

class IndicatorEngine:
    """Engine for calculating and caching technical indicators."""
    
    def __init__(self):
        self.cache: Dict[str, IndicatorResult] = {}
        self.cache_ttl = 300  # 5 minutes
    
    def calculate_indicator(self, 
                          indicator_name: str,
                          data: Dict[str, List[float]],
                          **params) -> Optional[IndicatorResult]:
        """Calculate a technical indicator."""
        
        # Create cache key
        cache_key = f"{indicator_name}_{hash(str(data))}_{hash(str(params))}"
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            result = None
            
            if indicator_name.upper() == 'SMA':
                result = TechnicalIndicators.sma(
                    data['close'], 
                    params.get('period', 20)
                )
            
            elif indicator_name.upper() == 'EMA':
                result = TechnicalIndicators.ema(
                    data['close'], 
                    params.get('period', 20)
                )
            
            elif indicator_name.upper() == 'RSI':
                result = TechnicalIndicators.rsi(
                    data['close'], 
                    params.get('period', 14)
                )
            
            elif indicator_name.upper() == 'MACD':
                result = TechnicalIndicators.macd(
                    data['close'],
                    params.get('fast_period', 12),
                    params.get('slow_period', 26),
                    params.get('signal_period', 9)
                )
            
            elif indicator_name.upper() == 'BOLLINGER_BANDS':
                result = TechnicalIndicators.bollinger_bands(
                    data['close'],
                    params.get('period', 20),
                    params.get('std_dev', 2.0)
                )
            
            elif indicator_name.upper() == 'STOCHASTIC':
                result = TechnicalIndicators.stochastic(
                    data['high'], data['low'], data['close'],
                    params.get('k_period', 14),
                    params.get('d_period', 3)
                )
            
            elif indicator_name.upper() == 'ATR':
                result = TechnicalIndicators.atr(
                    data['high'], data['low'], data['close'],
                    params.get('period', 14)
                )
            
            elif indicator_name.upper() == 'WILLIAMS_R':
                result = TechnicalIndicators.williams_r(
                    data['high'], data['low'], data['close'],
                    params.get('period', 14)
                )
            
            elif indicator_name.upper() == 'CCI':
                result = TechnicalIndicators.cci(
                    data['high'], data['low'], data['close'],
                    params.get('period', 20)
                )
            
            elif indicator_name.upper() == 'OBV':
                result = TechnicalIndicators.obv(
                    data['close'], data['volume']
                )
            
            elif indicator_name.upper() == 'ADX':
                result = TechnicalIndicators.adx(
                    data['high'], data['low'], data['close'],
                    params.get('period', 14)
                )
            
            if result:
                self.cache[cache_key] = result
                return result
        
        except Exception as e:
            logger.error(f"Error calculating {indicator_name}: {e}")
        
        return None
    
    def get_multiple_indicators(self, 
                               indicators: List[Dict[str, Any]], 
                               data: Dict[str, List[float]]) -> Dict[str, IndicatorResult]:
        """Calculate multiple indicators at once."""
        results = {}
        
        for indicator_config in indicators:
            name = indicator_config['name']
            params = indicator_config.get('params', {})
            
            result = self.calculate_indicator(name, data, **params)
            if result:
                results[name] = result
        
        return results
    
    def clear_cache(self):
        """Clear indicator cache."""
        self.cache.clear()
    
    def get_signal_analysis(self, 
                           symbol: str,
                           data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze multiple indicators to generate trading signals."""
        
        # Calculate key indicators
        indicators = [
            {'name': 'SMA', 'params': {'period': 20}},
            {'name': 'SMA', 'params': {'period': 50}},
            {'name': 'RSI', 'params': {'period': 14}},
            {'name': 'MACD', 'params': {}},
            {'name': 'BOLLINGER_BANDS', 'params': {}},
            {'name': 'STOCHASTIC', 'params': {}}
        ]
        
        results = self.get_multiple_indicators(indicators, data)
        
        if not results or not data.get('close'):
            return {'signals': [], 'overall_sentiment': 'neutral'}
        
        current_price = data['close'][-1]
        signals = []
        
        # SMA signals
        if 'SMA' in results:
            sma20 = results['SMA'].values[-1] if results['SMA'].values else 0
            if current_price > sma20:
                signals.append({'type': 'bullish', 'indicator': 'SMA20', 'strength': 0.3})
            elif current_price < sma20:
                signals.append({'type': 'bearish', 'indicator': 'SMA20', 'strength': 0.3})
        
        # RSI signals
        if 'RSI' in results:
            rsi = results['RSI'].values[-1] if results['RSI'].values else 50
            if rsi > 70:
                signals.append({'type': 'bearish', 'indicator': 'RSI_Overbought', 'strength': 0.4})
            elif rsi < 30:
                signals.append({'type': 'bullish', 'indicator': 'RSI_Oversold', 'strength': 0.4})
        
        # MACD signals
        if 'MACD' in results:
            macd_line = results['MACD'].values[-1] if results['MACD'].values else 0
            signal_line = results['MACD'].metadata.get('signal_line', [0])[-1]
            
            if macd_line > signal_line:
                signals.append({'type': 'bullish', 'indicator': 'MACD_Bullish', 'strength': 0.4})
            elif macd_line < signal_line:
                signals.append({'type': 'bearish', 'indicator': 'MACD_Bearish', 'strength': 0.4})
        
        # Calculate overall sentiment
        bullish_strength = sum(s['strength'] for s in signals if s['type'] == 'bullish')
        bearish_strength = sum(s['strength'] for s in signals if s['type'] == 'bearish')
        
        if bullish_strength > bearish_strength + 0.2:
            overall_sentiment = 'bullish'
        elif bearish_strength > bullish_strength + 0.2:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'symbol': symbol,
            'signals': signals,
            'overall_sentiment': overall_sentiment,
            'confidence': abs(bullish_strength - bearish_strength),
            'timestamp': pd.Timestamp.now().isoformat()
        }