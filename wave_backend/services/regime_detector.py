"""
Regime Detector
Market regime detection and adaptive strategy selection using advanced ML techniques.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from ..config.settings import get_settings
from ..services.event_bus import EventBus

logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """Market regime definition."""
    regime_id: str
    name: str
    description: str
    characteristics: Dict[str, float]
    typical_duration_days: int
    transition_probability: Dict[str, float]  # Probabilities to other regimes
    
@dataclass
class RegimeDetectionResult:
    """Result of regime detection analysis."""
    timestamp: datetime
    current_regime: str
    regime_probability: float
    regime_strength: float  # How strongly the current data fits the regime
    
    # Regime history and transitions
    regime_duration_days: int
    last_regime_change: Optional[datetime]
    transition_probability: Dict[str, float]
    
    # Market characteristics
    volatility_level: str  # 'low', 'medium', 'high', 'extreme'
    trend_strength: float  # -1 (strong downtrend) to 1 (strong uptrend)
    mean_reversion_strength: float  # 0-1, higher = more mean-reverting
    
    # Feature values used for classification
    features: Dict[str, float]
    
    # Confidence metrics
    classification_confidence: float
    stability_score: float  # How stable the regime has been

@dataclass
class AdaptiveRecommendation:
    """Adaptive strategy recommendation based on regime."""
    regime: str
    recommended_strategies: List[Dict]
    strategy_allocations: Dict[str, float]
    risk_adjustments: Dict[str, float]
    
    # Reasoning
    rationale: str
    confidence: float
    expected_regime_duration: int  # Expected days in this regime
    
    # Parameter adjustments
    parameter_modifications: Dict[str, Dict[str, Any]]

@dataclass
class VolatilityRegimeAnalysis:
    """Volatility regime analysis."""
    current_vol_regime: str  # 'low_vol', 'normal_vol', 'high_vol', 'crisis_vol'
    volatility_percentile: float  # Current vol vs historical distribution
    vol_clustering_strength: float  # Evidence of volatility clustering
    
    # GARCH-like analysis
    conditional_volatility: float
    vol_persistence: float  # How persistent current vol level is
    
    # Risk metrics
    tail_risk_elevated: bool
    correlation_breakdown: bool

class RegimeDetector:
    """Market regime detection and adaptive strategy selection."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.settings = get_settings()
        
        # Model settings
        self.lookback_days = getattr(self.settings.analytics, 'regime_detection_lookback_days', 252)
        self.n_regimes = 4  # trending, mean_reverting, high_vol, low_vol
        self.feature_window = 20  # Days for feature calculation
        
        # ML models
        self.regime_classifier = None
        self.volatility_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=6)
        
        # Regime definitions
        self.regimes = self._define_market_regimes()
        
        # Current state
        self.current_regime = None
        self.regime_history = []
        self.last_detection = None
        self.model_trained = False
        
        # Feature cache
        self.feature_cache = {}
        self.price_data_cache = {}
        
    async def detect_current_regime(self, symbols: List[str] = None) -> RegimeDetectionResult:
        """Detect current market regime."""
        
        if symbols is None:
            symbols = ['BTC/USDT', 'ETH/USDT']  # Default symbols
        
        logger.info("Detecting current market regime...")
        
        # Get market data
        market_data = await self._get_market_data(symbols)
        
        if len(market_data) < self.feature_window:
            logger.warning("Insufficient data for regime detection")
            return self._default_regime_result()
        
        # Extract features
        features = await self._extract_regime_features(market_data)
        
        # Ensure model is trained
        if not self.model_trained:
            await self._train_regime_models(market_data)
        
        # Classify current regime
        regime_prediction = await self._classify_regime(features)
        
        # Calculate regime characteristics
        regime_analysis = await self._analyze_regime_characteristics(market_data, features)
        
        # Update regime history
        await self._update_regime_history(regime_prediction['regime'])
        
        # Create result
        result = RegimeDetectionResult(
            timestamp=datetime.utcnow(),
            current_regime=regime_prediction['regime'],
            regime_probability=regime_prediction['probability'],
            regime_strength=regime_prediction['strength'],
            regime_duration_days=self._get_current_regime_duration(),
            last_regime_change=self._get_last_regime_change(),
            transition_probability=regime_prediction['transition_probs'],
            volatility_level=regime_analysis['volatility_level'],
            trend_strength=regime_analysis['trend_strength'],
            mean_reversion_strength=regime_analysis['mean_reversion_strength'],
            features=features,
            classification_confidence=regime_prediction['confidence'],
            stability_score=await self._calculate_stability_score()
        )
        
        self.last_detection = result
        
        # Emit regime change event if needed
        if (self.current_regime and 
            self.current_regime != result.current_regime and 
            result.regime_probability > 0.7):
            
            await self.event_bus.publish("regime_change", {
                "old_regime": self.current_regime,
                "new_regime": result.current_regime,
                "probability": result.regime_probability,
                "timestamp": result.timestamp.isoformat()
            })
            
            logger.info(f"Regime change detected: {self.current_regime} -> {result.current_regime}")
        
        self.current_regime = result.current_regime
        
        return result
    
    async def adaptive_parameter_adjustment(self, 
                                          strategy_id: str,
                                          current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically adjust parameters based on current regime."""
        
        if not self.last_detection:
            await self.detect_current_regime()
        
        current_regime = self.last_detection.current_regime
        regime_strength = self.last_detection.regime_strength
        
        logger.info(f"Adapting parameters for strategy {strategy_id} in {current_regime} regime")
        
        # Get base adjustments for regime
        regime_adjustments = self._get_regime_parameter_adjustments(current_regime)
        
        # Apply strength-based scaling
        adjusted_params = current_params.copy()
        
        for param_name, adjustment in regime_adjustments.items():
            if param_name in adjusted_params:
                base_value = adjusted_params[param_name]
                
                if isinstance(adjustment, dict):
                    # Complex adjustment with scaling
                    multiplier = adjustment.get('multiplier', 1.0)
                    offset = adjustment.get('offset', 0.0)
                    
                    # Scale by regime strength
                    scaled_multiplier = 1.0 + (multiplier - 1.0) * regime_strength
                    scaled_offset = offset * regime_strength
                    
                    new_value = base_value * scaled_multiplier + scaled_offset
                    
                    # Apply bounds if specified
                    if 'min_value' in adjustment:
                        new_value = max(new_value, adjustment['min_value'])
                    if 'max_value' in adjustment:
                        new_value = min(new_value, adjustment['max_value'])
                    
                    adjusted_params[param_name] = new_value
                
                else:
                    # Simple multiplier
                    adjusted_params[param_name] = base_value * adjustment
        
        return adjusted_params
    
    async def regime_specific_strategies(self) -> AdaptiveRecommendation:
        """Recommend different strategies for different regimes."""
        
        if not self.last_detection:
            await self.detect_current_regime()
        
        current_regime = self.last_detection.current_regime
        regime_prob = self.last_detection.regime_probability
        
        # Get regime-specific strategy recommendations
        recommendations = self._get_regime_strategy_recommendations(current_regime)
        
        # Calculate strategy allocations based on regime confidence
        allocations = self._calculate_regime_allocations(recommendations, regime_prob)
        
        # Calculate risk adjustments
        risk_adjustments = self._calculate_regime_risk_adjustments(current_regime)
        
        # Generate parameter modifications
        param_modifications = self._generate_parameter_modifications(current_regime)
        
        return AdaptiveRecommendation(
            regime=current_regime,
            recommended_strategies=recommendations['strategies'],
            strategy_allocations=allocations,
            risk_adjustments=risk_adjustments,
            rationale=recommendations['rationale'],
            confidence=regime_prob,
            expected_regime_duration=self.regimes[current_regime].typical_duration_days,
            parameter_modifications=param_modifications
        )
    
    async def volatility_regime_analysis(self) -> VolatilityRegimeAnalysis:
        """Analyze and predict volatility regimes."""
        
        logger.info("Analyzing volatility regime...")
        
        # Get market data
        market_data = await self._get_market_data(['BTC/USDT'])
        
        if len(market_data) < 60:  # Need at least 60 days
            return self._default_volatility_analysis()
        
        # Calculate returns and volatility
        returns = market_data['returns'].values
        realized_vol = self._calculate_realized_volatility(returns)
        
        # Classify volatility regime
        vol_regime = self._classify_volatility_regime(realized_vol)
        
        # Calculate volatility percentile
        vol_history = [self._calculate_realized_volatility(returns[max(0, i-20):i+1]) 
                      for i in range(20, len(returns))]
        vol_percentile = stats.percentileofscore(vol_history, realized_vol) / 100
        
        # Analyze volatility clustering
        vol_clustering = self._measure_volatility_clustering(returns)
        
        # Calculate conditional volatility (GARCH-like)
        conditional_vol = self._estimate_conditional_volatility(returns)
        
        # Calculate volatility persistence
        vol_persistence = self._calculate_volatility_persistence(returns)
        
        # Risk assessments
        tail_risk = self._assess_tail_risk(returns)
        correlation_breakdown = await self._detect_correlation_breakdown()
        
        return VolatilityRegimeAnalysis(
            current_vol_regime=vol_regime,
            volatility_percentile=vol_percentile,
            vol_clustering_strength=vol_clustering,
            conditional_volatility=conditional_vol,
            vol_persistence=vol_persistence,
            tail_risk_elevated=tail_risk,
            correlation_breakdown=correlation_breakdown
        )
    
    # Private implementation methods
    
    def _define_market_regimes(self) -> Dict[str, MarketRegime]:
        """Define the market regimes we can detect."""
        
        return {
            'trending_bull': MarketRegime(
                regime_id='trending_bull',
                name='Trending Bull Market',
                description='Strong upward trends with momentum',
                characteristics={
                    'trend_strength': 0.7,
                    'volatility': 0.4,
                    'mean_reversion': 0.2,
                    'momentum': 0.8
                },
                typical_duration_days=45,
                transition_probability={
                    'trending_bear': 0.15,
                    'mean_reverting': 0.25,
                    'high_volatility': 0.35,
                    'low_volatility': 0.25
                }
            ),
            'trending_bear': MarketRegime(
                regime_id='trending_bear',
                name='Trending Bear Market',
                description='Strong downward trends with momentum',
                characteristics={
                    'trend_strength': -0.7,
                    'volatility': 0.6,
                    'mean_reversion': 0.2,
                    'momentum': 0.8
                },
                typical_duration_days=30,
                transition_probability={
                    'trending_bull': 0.20,
                    'mean_reverting': 0.30,
                    'high_volatility': 0.40,
                    'low_volatility': 0.10
                }
            ),
            'mean_reverting': MarketRegime(
                regime_id='mean_reverting',
                name='Mean Reverting Market',
                description='Range-bound with mean reversion patterns',
                characteristics={
                    'trend_strength': 0.0,
                    'volatility': 0.3,
                    'mean_reversion': 0.8,
                    'momentum': 0.2
                },
                typical_duration_days=60,
                transition_probability={
                    'trending_bull': 0.30,
                    'trending_bear': 0.30,
                    'high_volatility': 0.25,
                    'low_volatility': 0.15
                }
            ),
            'high_volatility': MarketRegime(
                regime_id='high_volatility',
                name='High Volatility Market',
                description='Extreme volatility with unpredictable movements',
                characteristics={
                    'trend_strength': 0.1,
                    'volatility': 0.9,
                    'mean_reversion': 0.4,
                    'momentum': 0.3
                },
                typical_duration_days=20,
                transition_probability={
                    'trending_bull': 0.25,
                    'trending_bear': 0.25,
                    'mean_reverting': 0.35,
                    'low_volatility': 0.15
                }
            ),
            'low_volatility': MarketRegime(
                regime_id='low_volatility',
                name='Low Volatility Market',
                description='Stable, low volatility environment',
                characteristics={
                    'trend_strength': 0.2,
                    'volatility': 0.1,
                    'mean_reversion': 0.6,
                    'momentum': 0.3
                },
                typical_duration_days=90,
                transition_probability={
                    'trending_bull': 0.35,
                    'trending_bear': 0.15,
                    'mean_reverting': 0.40,
                    'high_volatility': 0.10
                }
            )
        }
    
    async def _get_market_data(self, symbols: List[str], days: int = None) -> pd.DataFrame:
        """Get market data for regime detection."""
        
        if days is None:
            days = self.lookback_days
        
        cache_key = f"{'_'.join(symbols)}_{days}"
        
        if cache_key in self.price_data_cache:
            cache_entry = self.price_data_cache[cache_key]
            if (datetime.utcnow() - cache_entry['timestamp']).seconds < 300:  # 5 min cache
                return cache_entry['data']
        
        # Generate realistic mock data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Generate correlated price series for multiple symbols
        n_days = len(dates)
        base_returns = np.random.normal(0.001, 0.025, n_days)
        
        data = {'date': dates}
        
        for i, symbol in enumerate(symbols):
            # Add symbol-specific noise but maintain correlation
            symbol_returns = base_returns + np.random.normal(0, 0.01, n_days) * (i + 1) * 0.5
            
            # Create regime-like patterns
            if n_days > 60:
                # Add trending periods
                trend_start = n_days // 4
                trend_end = trend_start + 30
                symbol_returns[trend_start:trend_end] += 0.003  # Bull trend
                
                # Add high volatility period
                vol_start = 3 * n_days // 4
                vol_end = vol_start + 15
                symbol_returns[vol_start:vol_end] *= 2  # High vol
            
            prices = 100 * np.cumprod(1 + symbol_returns)
            
            data[f'{symbol}_price'] = prices
            data[f'{symbol}_return'] = symbol_returns
        
        # Calculate aggregate market metrics
        if len(symbols) > 1:
            avg_returns = np.mean([data[f'{s}_return'] for s in symbols], axis=0)
            data['market_return'] = avg_returns
        else:
            data['market_return'] = data[f'{symbols[0]}_return']
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        # Cache result
        self.price_data_cache[cache_key] = {
            'data': df,
            'timestamp': datetime.utcnow()
        }
        
        return df
    
    async def _extract_regime_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract features for regime classification."""
        
        returns = market_data['market_return'].values
        
        # Volatility features
        realized_vol = np.std(returns[-self.feature_window:]) * np.sqrt(252)
        vol_of_vol = np.std([np.std(returns[max(0, i-5):i+1]) for i in range(5, len(returns))])
        
        # Trend features
        prices = market_data.iloc[:, 0].values  # First price column
        
        # Linear trend strength
        x = np.arange(len(prices[-self.feature_window:]))
        trend_slope, _, r_value, _, _ = stats.linregress(x, prices[-self.feature_window:])
        trend_strength = r_value ** 2  # R-squared
        trend_direction = 1 if trend_slope > 0 else -1
        
        # Moving average trends
        ma_5 = np.mean(prices[-5:])
        ma_20 = np.mean(prices[-20:])
        ma_trend = (ma_5 - ma_20) / ma_20
        
        # Mean reversion features
        price_zscore = (prices[-1] - np.mean(prices[-self.feature_window:])) / np.std(prices[-self.feature_window:])
        
        # Autocorrelation
        return_autocorr = self._calculate_autocorrelation(returns[-self.feature_window:])
        
        # Momentum features
        momentum_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 5 else 0
        momentum_20 = (prices[-1] - prices[-21]) / prices[-21] if len(prices) > 20 else 0
        
        # Volatility clustering
        vol_clustering = self._measure_volatility_clustering(returns[-self.feature_window:])
        
        # Skewness and kurtosis
        return_skew = stats.skew(returns[-self.feature_window:])
        return_kurt = stats.kurtosis(returns[-self.feature_window:])
        
        # Range-based metrics
        high_low_ratio = (max(prices[-self.feature_window:]) - min(prices[-self.feature_window:])) / np.mean(prices[-self.feature_window:])
        
        features = {
            'realized_volatility': realized_vol,
            'volatility_of_volatility': vol_of_vol,
            'trend_strength': trend_strength * trend_direction,
            'ma_trend': ma_trend,
            'price_zscore': price_zscore,
            'return_autocorr': return_autocorr,
            'momentum_5d': momentum_5,
            'momentum_20d': momentum_20,
            'volatility_clustering': vol_clustering,
            'return_skewness': return_skew,
            'return_kurtosis': return_kurt,
            'high_low_ratio': high_low_ratio,
            'vol_regime_indicator': 1 if realized_vol > 0.3 else 0,
            'trend_regime_indicator': 1 if abs(trend_strength * trend_direction) > 0.5 else 0
        }
        
        return features
    
    async def _train_regime_models(self, market_data: pd.DataFrame):
        """Train regime detection models."""
        
        logger.info("Training regime detection models...")
        
        # Generate training features for multiple time windows
        feature_matrix = []
        regime_labels = []
        
        min_window = max(self.feature_window, 30)
        
        for i in range(min_window, len(market_data), 5):  # Every 5 days
            window_data = market_data.iloc[max(0, i-self.lookback_days):i]
            
            if len(window_data) >= min_window:
                features = await self._extract_regime_features(window_data)
                
                # Generate synthetic regime label based on characteristics
                regime_label = self._synthetic_regime_labeling(features)
                
                feature_matrix.append(list(features.values()))
                regime_labels.append(regime_label)
        
        if len(feature_matrix) < 10:
            logger.warning("Insufficient training data for regime detection")
            return
        
        feature_matrix = np.array(feature_matrix)
        
        # Scale features
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        # Apply PCA for dimensionality reduction
        feature_matrix = self.pca.fit_transform(feature_matrix)
        
        # Train Gaussian Mixture Model for regime detection
        self.regime_classifier = GaussianMixture(
            n_components=len(self.regimes),
            covariance_type='full',
            random_state=42
        )
        
        self.regime_classifier.fit(feature_matrix)
        
        # Train Random Forest for regime classification
        self.volatility_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        # Create volatility regime labels
        vol_labels = ['low_vol' if f[0] < 0.2 else 'high_vol' if f[0] > 0.5 else 'normal_vol' 
                     for f in feature_matrix]
        
        if len(set(vol_labels)) > 1:  # Need multiple classes
            self.volatility_model.fit(feature_matrix, vol_labels)
        
        self.model_trained = True
        logger.info("Regime detection models trained successfully")
    
    def _synthetic_regime_labeling(self, features: Dict[str, float]) -> str:
        """Generate synthetic regime labels based on feature characteristics."""
        
        vol = features.get('realized_volatility', 0)
        trend = features.get('trend_strength', 0)
        mean_rev = abs(features.get('price_zscore', 0))
        
        # High volatility regime
        if vol > 0.4:
            return 'high_volatility'
        
        # Low volatility regime
        elif vol < 0.15:
            return 'low_volatility'
        
        # Strong trend regimes
        elif trend > 0.3:
            return 'trending_bull'
        elif trend < -0.3:
            return 'trending_bear'
        
        # Mean reverting regime
        else:
            return 'mean_reverting'
    
    async def _classify_regime(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Classify current regime using trained models."""
        
        if not self.model_trained:
            return {
                'regime': 'mean_reverting',
                'probability': 0.5,
                'strength': 0.5,
                'confidence': 0.5,
                'transition_probs': {regime: 0.2 for regime in self.regimes}
            }
        
        # Prepare features
        feature_vector = np.array([list(features.values())])
        feature_vector = self.scaler.transform(feature_vector)
        feature_vector = self.pca.transform(feature_vector)
        
        # Get regime probabilities
        regime_probs = self.regime_classifier.predict_proba(feature_vector)[0]
        
        # Map to regime names
        regime_names = list(self.regimes.keys())
        regime_prob_dict = dict(zip(regime_names, regime_probs))
        
        # Find most likely regime
        best_regime = max(regime_prob_dict, key=regime_prob_dict.get)
        best_prob = regime_prob_dict[best_regime]
        
        # Calculate regime strength based on how well features match
        expected_characteristics = self.regimes[best_regime].characteristics
        regime_strength = self._calculate_regime_fit(features, expected_characteristics)
        
        # Classification confidence
        confidence = best_prob
        
        # Transition probabilities
        transition_probs = self.regimes[best_regime].transition_probability
        
        return {
            'regime': best_regime,
            'probability': best_prob,
            'strength': regime_strength,
            'confidence': confidence,
            'transition_probs': transition_probs
        }
    
    def _calculate_regime_fit(self, features: Dict[str, float], expected: Dict[str, float]) -> float:
        """Calculate how well current features fit expected regime characteristics."""
        
        # Map features to regime characteristics
        feature_mapping = {
            'trend_strength': features.get('trend_strength', 0),
            'volatility': min(features.get('realized_volatility', 0), 1.0),
            'mean_reversion': min(abs(features.get('price_zscore', 0)) / 2, 1.0),
            'momentum': (abs(features.get('momentum_5d', 0)) + abs(features.get('momentum_20d', 0))) / 2
        }
        
        # Calculate similarity
        similarities = []
        for char, expected_value in expected.items():
            if char in feature_mapping:
                actual_value = feature_mapping[char]
                similarity = 1.0 - abs(expected_value - actual_value)
                similarities.append(max(0, similarity))
        
        return np.mean(similarities) if similarities else 0.5
    
    async def _analyze_regime_characteristics(self, market_data: pd.DataFrame, features: Dict[str, float]) -> Dict[str, Any]:
        """Analyze current market characteristics."""
        
        returns = market_data['market_return'].values
        
        # Volatility level
        vol = features.get('realized_volatility', 0)
        if vol < 0.15:
            vol_level = 'low'
        elif vol < 0.25:
            vol_level = 'medium'  
        elif vol < 0.40:
            vol_level = 'high'
        else:
            vol_level = 'extreme'
        
        # Trend strength
        trend_strength = features.get('trend_strength', 0)
        
        # Mean reversion strength
        mean_reversion_strength = min(abs(features.get('price_zscore', 0)) / 3, 1.0)
        
        return {
            'volatility_level': vol_level,
            'trend_strength': trend_strength,
            'mean_reversion_strength': mean_reversion_strength
        }
    
    async def _update_regime_history(self, regime: str):
        """Update regime history with new detection."""
        
        current_time = datetime.utcnow()
        
        # Add to history if regime changed
        if not self.regime_history or self.regime_history[-1]['regime'] != regime:
            self.regime_history.append({
                'regime': regime,
                'start_time': current_time,
                'detection_time': current_time
            })
            
            # Limit history size
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
    
    def _get_current_regime_duration(self) -> int:
        """Get duration of current regime in days."""
        
        if not self.regime_history:
            return 0
        
        current_regime_start = self.regime_history[-1]['start_time']
        duration = (datetime.utcnow() - current_regime_start).days
        
        return max(duration, 0)
    
    def _get_last_regime_change(self) -> Optional[datetime]:
        """Get timestamp of last regime change."""
        
        if len(self.regime_history) < 2:
            return None
        
        return self.regime_history[-1]['start_time']
    
    async def _calculate_stability_score(self) -> float:
        """Calculate stability score of current regime."""
        
        if len(self.regime_history) < 2:
            return 1.0
        
        # Look at regime changes in last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_changes = [
            h for h in self.regime_history 
            if h['start_time'] > thirty_days_ago
        ]
        
        # More changes = less stability
        change_count = len(recent_changes)
        stability = max(0, 1.0 - (change_count * 0.2))
        
        return stability
    
    # Utility methods for feature calculation
    
    def _calculate_autocorrelation(self, returns: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation of returns."""
        
        if len(returns) <= lag:
            return 0.0
        
        return np.corrcoef(returns[:-lag], returns[lag:])[0, 1] if len(returns) > lag else 0.0
    
    def _measure_volatility_clustering(self, returns: np.ndarray) -> float:
        """Measure strength of volatility clustering."""
        
        if len(returns) < 10:
            return 0.0
        
        # Calculate absolute returns
        abs_returns = np.abs(returns)
        
        # Autocorrelation of absolute returns indicates clustering
        clustering = self._calculate_autocorrelation(abs_returns, lag=1)
        
        return max(0, clustering) if not np.isnan(clustering) else 0.0
    
    def _calculate_realized_volatility(self, returns: np.ndarray, annualize: bool = True) -> float:
        """Calculate realized volatility."""
        
        if len(returns) < 2:
            return 0.0
        
        vol = np.std(returns)
        
        if annualize:
            vol *= np.sqrt(252)  # Annualize assuming daily data
        
        return vol
    
    def _classify_volatility_regime(self, realized_vol: float) -> str:
        """Classify volatility into regime categories."""
        
        if realized_vol < 0.15:
            return 'low_vol'
        elif realized_vol < 0.25:
            return 'normal_vol'
        elif realized_vol < 0.50:
            return 'high_vol'
        else:
            return 'crisis_vol'
    
    def _estimate_conditional_volatility(self, returns: np.ndarray) -> float:
        """Estimate conditional volatility (simple GARCH-like)."""
        
        if len(returns) < 20:
            return np.std(returns) if len(returns) > 1 else 0.0
        
        # Simple EWMA model
        lambda_param = 0.94
        
        # Initialize with sample variance
        conditional_var = np.var(returns[:20])
        
        # Update with EWMA
        for i in range(20, len(returns)):
            conditional_var = lambda_param * conditional_var + (1 - lambda_param) * returns[i] ** 2
        
        return np.sqrt(conditional_var * 252)  # Annualized
    
    def _calculate_volatility_persistence(self, returns: np.ndarray) -> float:
        """Calculate volatility persistence."""
        
        if len(returns) < 30:
            return 0.5
        
        # Calculate rolling volatilities
        window = 10
        vol_series = [np.std(returns[max(0, i-window):i+1]) 
                     for i in range(window, len(returns))]
        
        # Autocorrelation of volatility series
        persistence = self._calculate_autocorrelation(np.array(vol_series))
        
        return max(0, persistence) if not np.isnan(persistence) else 0.5
    
    def _assess_tail_risk(self, returns: np.ndarray) -> bool:
        """Assess if tail risk is elevated."""
        
        if len(returns) < 30:
            return False
        
        # Check for extreme returns in recent period
        recent_returns = returns[-20:]
        
        # Calculate VaR and check for exceedances
        var_95 = np.percentile(returns, 5)
        exceedances = sum(1 for r in recent_returns if r < var_95)
        
        # More than expected exceedances indicates elevated tail risk
        expected_exceedances = len(recent_returns) * 0.05
        
        return exceedances > expected_exceedances * 2
    
    async def _detect_correlation_breakdown(self) -> bool:
        """Detect if correlations are breaking down."""
        
        # Mock implementation - would analyze cross-asset correlations
        # In practice, would look at rolling correlations between assets
        
        # Random simulation of correlation breakdown
        return np.random.random() < 0.1  # 10% chance of breakdown
    
    # Adaptive strategy methods
    
    def _get_regime_parameter_adjustments(self, regime: str) -> Dict[str, Any]:
        """Get parameter adjustments for specific regime."""
        
        adjustments = {
            'trending_bull': {
                'stop_loss': {'multiplier': 0.8, 'min_value': 0.005},  # Looser stops in trends
                'take_profit': {'multiplier': 1.5, 'max_value': 0.10},  # Higher targets
                'position_size': {'multiplier': 1.2, 'max_value': 0.25},  # Larger positions
                'rsi_period': {'multiplier': 0.8, 'min_value': 10},  # Shorter periods
                'sma_fast': {'multiplier': 0.9, 'min_value': 5}
            },
            'trending_bear': {
                'stop_loss': {'multiplier': 0.7, 'min_value': 0.005},  # Tight stops
                'take_profit': {'multiplier': 1.0},  # Standard targets
                'position_size': {'multiplier': 0.8, 'min_value': 0.05},  # Smaller positions
                'rsi_period': {'multiplier': 0.8, 'min_value': 10},
                'sma_fast': {'multiplier': 0.9, 'min_value': 5}
            },
            'mean_reverting': {
                'stop_loss': {'multiplier': 1.3, 'max_value': 0.03},  # Wider stops for reversals
                'take_profit': {'multiplier': 0.8, 'min_value': 0.01},  # Quick profits
                'position_size': {'multiplier': 1.0},  # Standard size
                'rsi_period': {'multiplier': 1.2, 'max_value': 25},  # Longer periods
                'sma_fast': {'multiplier': 1.1, 'max_value': 30}
            },
            'high_volatility': {
                'stop_loss': {'multiplier': 1.5, 'max_value': 0.05},  # Wide stops
                'take_profit': {'multiplier': 0.6, 'min_value': 0.01},  # Quick profits
                'position_size': {'multiplier': 0.6, 'min_value': 0.03},  # Small positions
                'rsi_period': {'multiplier': 0.7, 'min_value': 8},  # Reactive
                'order_frequency_limit': {'multiplier': 0.5}  # Less frequent trading
            },
            'low_volatility': {
                'stop_loss': {'multiplier': 0.9},  # Standard stops
                'take_profit': {'multiplier': 1.2, 'max_value': 0.08},  # Patient profits
                'position_size': {'multiplier': 1.1, 'max_value': 0.30},  # Larger positions
                'rsi_period': {'multiplier': 1.3, 'max_value': 30},  # Patient signals
                'order_frequency_limit': {'multiplier': 1.2}  # More frequent trading
            }
        }
        
        return adjustments.get(regime, {})
    
    def _get_regime_strategy_recommendations(self, regime: str) -> Dict[str, Any]:
        """Get strategy recommendations for regime."""
        
        recommendations = {
            'trending_bull': {
                'strategies': [
                    {'type': 'momentum', 'weight': 0.4, 'priority': 1},
                    {'type': 'trend_following', 'weight': 0.4, 'priority': 1},
                    {'type': 'breakout', 'weight': 0.2, 'priority': 2}
                ],
                'rationale': 'Bull trend favors momentum and trend-following strategies'
            },
            'trending_bear': {
                'strategies': [
                    {'type': 'short_momentum', 'weight': 0.3, 'priority': 1},
                    {'type': 'mean_reversion', 'weight': 0.4, 'priority': 1},
                    {'type': 'defensive', 'weight': 0.3, 'priority': 2}
                ],
                'rationale': 'Bear trend requires defensive positioning and mean reversion'
            },
            'mean_reverting': {
                'strategies': [
                    {'type': 'mean_reversion', 'weight': 0.5, 'priority': 1},
                    {'type': 'pairs_trading', 'weight': 0.3, 'priority': 2},
                    {'type': 'range_trading', 'weight': 0.2, 'priority': 2}
                ],
                'rationale': 'Range-bound market favors mean reversion strategies'
            },
            'high_volatility': {
                'strategies': [
                    {'type': 'volatility_trading', 'weight': 0.4, 'priority': 1},
                    {'type': 'defensive', 'weight': 0.4, 'priority': 1},
                    {'type': 'momentum', 'weight': 0.2, 'priority': 3}
                ],
                'rationale': 'High volatility requires defensive positioning and vol strategies'
            },
            'low_volatility': {
                'strategies': [
                    {'type': 'carry_trade', 'weight': 0.3, 'priority': 1},
                    {'type': 'trend_following', 'weight': 0.3, 'priority': 1},
                    {'type': 'momentum', 'weight': 0.4, 'priority': 2}
                ],
                'rationale': 'Low volatility environment supports carry and trend strategies'
            }
        }
        
        return recommendations.get(regime, recommendations['mean_reverting'])
    
    def _calculate_regime_allocations(self, 
                                    recommendations: Dict[str, Any], 
                                    regime_confidence: float) -> Dict[str, float]:
        """Calculate strategy allocations based on regime confidence."""
        
        strategies = recommendations['strategies']
        
        # Base allocations from recommendations
        allocations = {}
        
        for strategy in strategies:
            strategy_type = strategy['type']
            base_weight = strategy['weight']
            
            # Adjust weight by regime confidence
            adjusted_weight = base_weight * regime_confidence + (base_weight * 0.5) * (1 - regime_confidence)
            
            allocations[strategy_type] = adjusted_weight
        
        # Normalize to sum to 1
        total_weight = sum(allocations.values())
        if total_weight > 0:
            allocations = {k: v / total_weight for k, v in allocations.items()}
        
        return allocations
    
    def _calculate_regime_risk_adjustments(self, regime: str) -> Dict[str, float]:
        """Calculate risk adjustments for regime."""
        
        risk_adjustments = {
            'trending_bull': {
                'position_size_multiplier': 1.1,
                'stop_loss_multiplier': 0.9,
                'leverage_multiplier': 1.2
            },
            'trending_bear': {
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 0.8,
                'leverage_multiplier': 0.7
            },
            'mean_reverting': {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.1,
                'leverage_multiplier': 1.0
            },
            'high_volatility': {
                'position_size_multiplier': 0.6,
                'stop_loss_multiplier': 1.4,
                'leverage_multiplier': 0.5
            },
            'low_volatility': {
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 0.9,
                'leverage_multiplier': 1.1
            }
        }
        
        return risk_adjustments.get(regime, risk_adjustments['mean_reverting'])
    
    def _generate_parameter_modifications(self, regime: str) -> Dict[str, Dict[str, Any]]:
        """Generate parameter modifications for different strategies."""
        
        modifications = {
            'sma_crossover': self._get_regime_parameter_adjustments(regime),
            'rsi_mean_reversion': self._get_regime_parameter_adjustments(regime),
            'momentum_strategy': self._get_regime_parameter_adjustments(regime)
        }
        
        return modifications
    
    # Default/fallback methods
    
    def _default_regime_result(self) -> RegimeDetectionResult:
        """Return default regime detection result."""
        
        return RegimeDetectionResult(
            timestamp=datetime.utcnow(),
            current_regime='mean_reverting',
            regime_probability=0.5,
            regime_strength=0.5,
            regime_duration_days=0,
            last_regime_change=None,
            transition_probability={regime: 0.2 for regime in self.regimes},
            volatility_level='medium',
            trend_strength=0.0,
            mean_reversion_strength=0.5,
            features={},
            classification_confidence=0.5,
            stability_score=0.5
        )
    
    def _default_volatility_analysis(self) -> VolatilityRegimeAnalysis:
        """Return default volatility analysis."""
        
        return VolatilityRegimeAnalysis(
            current_vol_regime='normal_vol',
            volatility_percentile=0.5,
            vol_clustering_strength=0.3,
            conditional_volatility=0.2,
            vol_persistence=0.5,
            tail_risk_elevated=False,
            correlation_breakdown=False
        )
    
    def get_regime_status(self) -> Dict[str, Any]:
        """Get current regime detector status."""
        
        return {
            'model_trained': self.model_trained,
            'current_regime': self.current_regime,
            'last_detection': self.last_detection.timestamp.isoformat() if self.last_detection else None,
            'regime_history_count': len(self.regime_history),
            'available_regimes': list(self.regimes.keys()),
            'feature_cache_size': len(self.feature_cache)
        }