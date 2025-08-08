"""
Portfolio Optimization
Advanced portfolio optimization using Modern Portfolio Theory, Black-Litterman, and Risk Parity.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy import linalg
from sklearn.covariance import LedoitWolf

from ..config.settings import get_settings
from ..services.event_bus import EventBus

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Portfolio optimization result."""
    optimization_method: str
    strategy_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    
    # Risk metrics
    portfolio_var: float
    diversification_ratio: float
    concentration_index: float
    
    # Optimization details
    success: bool
    iterations: int
    optimization_time: float
    constraints_satisfied: bool
    
    # Performance attribution
    contribution_to_risk: Dict[str, float]
    contribution_to_return: Dict[str, float]

@dataclass
class RebalancingSignal:
    """Portfolio rebalancing signal."""
    timestamp: datetime
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    rebalancing_needs: Dict[str, float]  # Difference between target and current
    
    # Rebalancing metrics
    turnover_required: float
    transaction_costs_estimate: float
    expected_improvement: float
    
    # Urgency and triggers
    urgency_score: float  # 0-1, higher = more urgent
    triggers: List[str]  # Reasons for rebalancing
    
@dataclass
class BlackLittermanInputs:
    """Black-Litterman model inputs."""
    views: Dict[str, float]  # Expected returns for each strategy
    confidence_matrix: np.ndarray  # Confidence in each view
    tau: float  # Scaling factor for uncertainty
    risk_aversion: float  # Investor risk aversion

@dataclass
class RiskParityResult:
    """Risk parity optimization result."""
    weights: Dict[str, float]
    risk_contributions: Dict[str, float]
    risk_parity_error: float  # How close to equal risk contribution
    total_volatility: float
    diversification_benefit: float

class PortfolioOptimizer:
    """Advanced portfolio optimization using multiple methodologies."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.settings = get_settings()
        
        # Optimization settings
        self.max_weight = 0.4  # Maximum weight per strategy
        self.min_weight = 0.0  # Minimum weight per strategy  
        self.transaction_cost_bps = 5  # 5 basis points per trade
        
        # Risk model settings
        self.lookback_days = 252  # 1 year for covariance estimation
        self.min_strategies = 2
        self.max_strategies = 20
        
        # Rebalancing settings
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalancing
        self.min_rebalance_interval_days = 7  # Minimum 1 week between rebalances
        
        # Cache for optimization results
        self.optimization_cache = {}
        self.covariance_cache = {}
        
        # Current portfolio state
        self.current_weights = {}
        self.last_rebalance = None
        
    async def optimize_weights(self, 
                             strategies: List[Dict],
                             method: str = "max_sharpe",
                             constraints: Optional[Dict] = None) -> OptimizationResult:
        """Optimize portfolio allocation across strategies."""
        
        start_time = datetime.utcnow()
        logger.info(f"Starting portfolio optimization using {method} method")
        
        if len(strategies) < self.min_strategies:
            raise ValueError(f"Need at least {self.min_strategies} strategies for optimization")
        
        # Get expected returns and covariance matrix
        expected_returns = await self._estimate_expected_returns(strategies)
        covariance_matrix = await self._estimate_covariance_matrix(strategies)
        
        # Validate inputs
        if len(expected_returns) != covariance_matrix.shape[0]:
            raise ValueError("Dimension mismatch between returns and covariance matrix")
        
        # Set up optimization constraints
        opt_constraints = self._setup_optimization_constraints(strategies, constraints)
        
        # Run optimization based on method
        if method == "max_sharpe":
            result = await self._optimize_max_sharpe(expected_returns, covariance_matrix, opt_constraints)
        elif method == "min_variance":
            result = await self._optimize_min_variance(covariance_matrix, opt_constraints)
        elif method == "max_return":
            result = await self._optimize_max_return(expected_returns, opt_constraints)
        elif method == "equal_weight":
            result = await self._optimize_equal_weight(strategies)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(result['weights'], expected_returns)
        portfolio_var = np.dot(result['weights'], np.dot(covariance_matrix, result['weights']))
        portfolio_vol = np.sqrt(portfolio_var)
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Calculate additional metrics
        diversification_ratio = self._calculate_diversification_ratio(result['weights'], covariance_matrix)
        concentration_index = self._calculate_concentration_index(result['weights'])
        
        # Performance attribution
        contribution_to_risk = self._calculate_risk_contribution(result['weights'], covariance_matrix)
        contribution_to_return = {
            strategies[i]['id']: result['weights'][i] * expected_returns[i]
            for i in range(len(strategies))
        }
        
        # Create strategy weights dictionary
        strategy_weights = {
            strategies[i]['id']: result['weights'][i]
            for i in range(len(strategies))
        }
        
        optimization_time = (datetime.utcnow() - start_time).total_seconds()
        
        optimization_result = OptimizationResult(
            optimization_method=method,
            strategy_weights=strategy_weights,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe_ratio,
            portfolio_var=portfolio_var,
            diversification_ratio=diversification_ratio,
            concentration_index=concentration_index,
            success=result.get('success', True),
            iterations=result.get('iterations', 0),
            optimization_time=optimization_time,
            constraints_satisfied=result.get('constraints_satisfied', True),
            contribution_to_risk=contribution_to_risk,
            contribution_to_return=contribution_to_return
        )
        
        logger.info(f"Optimization completed in {optimization_time:.2f}s: "
                   f"Sharpe={sharpe_ratio:.3f}, Vol={portfolio_vol:.3f}")
        
        return optimization_result
    
    async def black_litterman_optimization(self, 
                                         strategies: List[Dict],
                                         bl_inputs: BlackLittermanInputs) -> OptimizationResult:
        """Apply Black-Litterman model with user views."""
        
        logger.info("Starting Black-Litterman optimization")
        
        # Get market equilibrium returns and covariance
        covariance_matrix = await self._estimate_covariance_matrix(strategies)
        market_caps = await self._estimate_market_caps(strategies)
        
        # Calculate implied equilibrium returns
        risk_aversion = bl_inputs.risk_aversion
        equilibrium_returns = risk_aversion * np.dot(covariance_matrix, market_caps)
        
        # Set up views matrix
        n_strategies = len(strategies)
        n_views = len(bl_inputs.views)
        
        P = np.zeros((n_views, n_strategies))  # Picking matrix
        Q = np.zeros(n_views)  # View returns
        
        # Map views to strategies
        strategy_map = {s['id']: i for i, s in enumerate(strategies)}
        
        for i, (strategy_id, view_return) in enumerate(bl_inputs.views.items()):
            if strategy_id in strategy_map:
                P[i, strategy_map[strategy_id]] = 1.0
                Q[i] = view_return
        
        # Black-Litterman calculation
        tau = bl_inputs.tau
        omega = bl_inputs.confidence_matrix
        
        # New expected returns
        inv_cov = linalg.inv(covariance_matrix)
        inv_omega = linalg.inv(omega)
        
        M1 = linalg.inv(tau * inv_cov + P.T @ inv_omega @ P)
        M2 = tau * inv_cov @ equilibrium_returns + P.T @ inv_omega @ Q
        
        bl_returns = M1 @ M2
        
        # New covariance matrix
        bl_covariance = M1
        
        # Optimize with Black-Litterman inputs
        constraints = self._setup_optimization_constraints(strategies, None)
        result = await self._optimize_max_sharpe(bl_returns, bl_covariance, constraints)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(result['weights'], bl_returns)
        portfolio_vol = np.sqrt(np.dot(result['weights'], np.dot(bl_covariance, result['weights'])))
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        strategy_weights = {
            strategies[i]['id']: result['weights'][i]
            for i in range(len(strategies))
        }
        
        return OptimizationResult(
            optimization_method="black_litterman",
            strategy_weights=strategy_weights,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe_ratio,
            portfolio_var=np.dot(result['weights'], np.dot(bl_covariance, result['weights'])),
            diversification_ratio=self._calculate_diversification_ratio(result['weights'], bl_covariance),
            concentration_index=self._calculate_concentration_index(result['weights']),
            success=result.get('success', True),
            iterations=result.get('iterations', 0),
            optimization_time=0.0,
            constraints_satisfied=True,
            contribution_to_risk=self._calculate_risk_contribution(result['weights'], bl_covariance),
            contribution_to_return={
                strategies[i]['id']: result['weights'][i] * bl_returns[i]
                for i in range(len(strategies))
            }
        )
    
    async def risk_parity_allocation(self, strategies: List[Dict]) -> RiskParityResult:
        """Risk parity portfolio construction."""
        
        logger.info("Starting risk parity optimization")
        
        # Get covariance matrix
        covariance_matrix = await self._estimate_covariance_matrix(strategies)
        n_assets = len(strategies)
        
        # Objective function: minimize sum of squared risk contribution deviations
        def risk_parity_objective(weights):
            weights = np.abs(weights)  # Ensure positive weights
            weights = weights / np.sum(weights)  # Normalize
            
            portfolio_var = np.dot(weights, np.dot(covariance_matrix, weights))
            
            if portfolio_var <= 0:
                return 1e6
            
            # Calculate risk contributions
            marginal_contrib = np.dot(covariance_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_var
            
            # Target: equal risk contribution (1/n each)
            target_contrib = 1.0 / n_assets
            
            # Sum of squared deviations from equal risk contribution
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]
        
        bounds = [(0.01, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            logger.warning("Risk parity optimization did not converge, using equal weights")
            optimal_weights = np.ones(n_assets) / n_assets
        else:
            optimal_weights = np.abs(result.x)
            optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # Calculate final risk contributions
        portfolio_var = np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights))
        portfolio_vol = np.sqrt(portfolio_var)
        
        marginal_contrib = np.dot(covariance_matrix, optimal_weights)
        risk_contributions = optimal_weights * marginal_contrib / portfolio_var
        
        # Calculate risk parity error
        target_contrib = 1.0 / n_assets
        risk_parity_error = np.sum((risk_contributions - target_contrib) ** 2)
        
        # Diversification benefit
        individual_vols = np.sqrt(np.diag(covariance_matrix))
        weighted_avg_vol = np.dot(optimal_weights, individual_vols)
        diversification_benefit = 1 - (portfolio_vol / weighted_avg_vol)
        
        strategy_weights = {
            strategies[i]['id']: optimal_weights[i]
            for i in range(len(strategies))
        }
        
        strategy_risk_contributions = {
            strategies[i]['id']: risk_contributions[i]
            for i in range(len(strategies))
        }
        
        return RiskParityResult(
            weights=strategy_weights,
            risk_contributions=strategy_risk_contributions,
            risk_parity_error=risk_parity_error,
            total_volatility=portfolio_vol,
            diversification_benefit=diversification_benefit
        )
    
    async def dynamic_rebalancing(self, 
                                current_portfolio: Dict[str, float],
                                target_weights: Dict[str, float],
                                market_conditions: Optional[Dict] = None) -> RebalancingSignal:
        """Dynamic portfolio rebalancing based on market conditions."""
        
        logger.info("Analyzing rebalancing needs")
        
        # Calculate current deviations
        rebalancing_needs = {}
        total_deviation = 0
        
        for strategy_id in target_weights:
            current_weight = current_portfolio.get(strategy_id, 0.0)
            target_weight = target_weights[strategy_id]
            deviation = target_weight - current_weight
            
            rebalancing_needs[strategy_id] = deviation
            total_deviation += abs(deviation)
        
        # Calculate turnover required
        turnover_required = total_deviation / 2  # Each deviation requires buying and selling
        
        # Estimate transaction costs
        transaction_costs = turnover_required * (self.transaction_cost_bps / 10000)
        
        # Determine urgency score
        urgency_score = self._calculate_rebalancing_urgency(
            rebalancing_needs, market_conditions
        )
        
        # Identify triggers
        triggers = self._identify_rebalancing_triggers(
            rebalancing_needs, market_conditions
        )
        
        # Estimate expected improvement from rebalancing
        expected_improvement = self._estimate_rebalancing_benefit(
            current_portfolio, target_weights
        )
        
        return RebalancingSignal(
            timestamp=datetime.utcnow(),
            current_weights=current_portfolio.copy(),
            target_weights=target_weights.copy(),
            rebalancing_needs=rebalancing_needs,
            turnover_required=turnover_required,
            transaction_costs_estimate=transaction_costs,
            expected_improvement=expected_improvement,
            urgency_score=urgency_score,
            triggers=triggers
        )
    
    # Optimization methods
    
    async def _optimize_max_sharpe(self, 
                                 expected_returns: np.ndarray,
                                 covariance_matrix: np.ndarray,
                                 constraints: List[Dict]) -> Dict:
        """Maximize Sharpe ratio optimization."""
        
        n_assets = len(expected_returns)
        
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_var = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_var)
            
            if portfolio_vol <= 0:
                return 1e6
            
            return -portfolio_return / portfolio_vol  # Minimize negative Sharpe
        
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        initial_guess = np.ones(n_assets) / n_assets
        
        result = minimize(
            negative_sharpe,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        return {
            'weights': result.x / np.sum(result.x),  # Ensure normalization
            'success': result.success,
            'iterations': result.nit if hasattr(result, 'nit') else 0,
            'constraints_satisfied': all([
                abs(constraint['fun'](result.x)) < 1e-6 
                for constraint in constraints
            ])
        }
    
    async def _optimize_min_variance(self, 
                                   covariance_matrix: np.ndarray,
                                   constraints: List[Dict]) -> Dict:
        """Minimum variance optimization."""
        
        n_assets = covariance_matrix.shape[0]
        
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))
        
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        initial_guess = np.ones(n_assets) / n_assets
        
        result = minimize(
            portfolio_variance,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        return {
            'weights': result.x / np.sum(result.x),
            'success': result.success,
            'iterations': result.nit if hasattr(result, 'nit') else 0,
            'constraints_satisfied': True
        }
    
    async def _optimize_max_return(self, 
                                 expected_returns: np.ndarray,
                                 constraints: List[Dict]) -> Dict:
        """Maximum return optimization."""
        
        n_assets = len(expected_returns)
        
        def negative_return(weights):
            return -np.dot(weights, expected_returns)
        
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        initial_guess = np.ones(n_assets) / n_assets
        
        result = minimize(
            negative_return,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        return {
            'weights': result.x / np.sum(result.x),
            'success': result.success,
            'iterations': result.nit if hasattr(result, 'nit') else 0,
            'constraints_satisfied': True
        }
    
    async def _optimize_equal_weight(self, strategies: List[Dict]) -> Dict:
        """Equal weight allocation."""
        n_assets = len(strategies)
        weights = np.ones(n_assets) / n_assets
        
        return {
            'weights': weights,
            'success': True,
            'iterations': 0,
            'constraints_satisfied': True
        }
    
    # Helper methods
    
    async def _estimate_expected_returns(self, strategies: List[Dict]) -> np.ndarray:
        """Estimate expected returns for strategies."""
        expected_returns = []
        
        for strategy in strategies:
            # Get historical performance
            perf_data = await self._get_strategy_performance(strategy['id'])
            
            if perf_data and len(perf_data['returns']) > 0:
                # Use historical mean with some shrinkage
                historical_mean = np.mean(perf_data['returns'])
                
                # Shrink towards grand mean (James-Stein estimator concept)
                grand_mean = 0.0005  # ~12% annual return assumption
                shrinkage = 0.3
                
                expected_return = (1 - shrinkage) * historical_mean + shrinkage * grand_mean
                expected_returns.append(expected_return)
            else:
                # Default assumption
                expected_returns.append(0.0005)  # ~12% annual return
        
        return np.array(expected_returns)
    
    async def _estimate_covariance_matrix(self, strategies: List[Dict]) -> np.ndarray:
        """Estimate covariance matrix using robust estimation."""
        
        cache_key = f"cov_{'_'.join([s['id'] for s in strategies])}"
        if cache_key in self.covariance_cache:
            cache_entry = self.covariance_cache[cache_key]
            if (datetime.utcnow() - cache_entry['timestamp']).seconds < 3600:  # 1 hour cache
                return cache_entry['matrix']
        
        # Collect return series
        return_series = []
        
        for strategy in strategies:
            perf_data = await self._get_strategy_performance(strategy['id'])
            
            if perf_data and len(perf_data['returns']) >= 30:
                returns = np.array(perf_data['returns'][-self.lookback_days:])  # Use recent data
                return_series.append(returns)
            else:
                # Generate synthetic returns based on strategy type
                returns = self._generate_synthetic_returns(strategy)
                return_series.append(returns)
        
        # Align series lengths
        min_length = min(len(series) for series in return_series)
        aligned_series = [series[-min_length:] for series in return_series]
        
        # Create returns matrix
        returns_matrix = np.column_stack(aligned_series)
        
        # Use Ledoit-Wolf shrinkage estimator for robust covariance
        cov_estimator = LedoitWolf()
        covariance_matrix = cov_estimator.fit(returns_matrix).covariance_
        
        # Cache result
        self.covariance_cache[cache_key] = {
            'matrix': covariance_matrix,
            'timestamp': datetime.utcnow()
        }
        
        return covariance_matrix
    
    def _generate_synthetic_returns(self, strategy: Dict, n_days: int = 252) -> np.ndarray:
        """Generate synthetic returns for strategies without sufficient history."""
        
        # Base parameters by strategy type
        strategy_params = {
            'trend': {'mean': 0.0008, 'vol': 0.020},  # Trend following
            'mean_reversion': {'mean': 0.0006, 'vol': 0.015},  # Mean reversion
            'momentum': {'mean': 0.0010, 'vol': 0.025},  # Momentum
            'arbitrage': {'mean': 0.0004, 'vol': 0.008},  # Arbitrage
        }
        
        strategy_type = strategy.get('type', 'trend')
        params = strategy_params.get(strategy_type, strategy_params['trend'])
        
        # Generate returns with some autocorrelation
        np.random.seed(hash(strategy['id']) % 2**32)  # Deterministic per strategy
        
        # Base returns
        returns = np.random.normal(params['mean'], params['vol'], n_days)
        
        # Add some autocorrelation
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]  # Small momentum effect
        
        return returns
    
    async def _estimate_market_caps(self, strategies: List[Dict]) -> np.ndarray:
        """Estimate market capitalizations for equilibrium weights."""
        # For strategy portfolios, use AUM or equal weights
        market_caps = []
        
        for strategy in strategies:
            aum = strategy.get('aum', 1000000)  # Default $1M AUM
            market_caps.append(aum)
        
        market_caps = np.array(market_caps)
        return market_caps / np.sum(market_caps)  # Normalize to weights
    
    async def _get_strategy_performance(self, strategy_id: str) -> Optional[Dict]:
        """Get strategy performance data."""
        # Mock implementation - would fetch from database/service
        cache_key = f"perf_{strategy_id}"
        
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
        
        # Generate mock performance data
        np.random.seed(hash(strategy_id) % 2**32)
        
        n_days = 252
        daily_returns = np.random.normal(0.0008, 0.018, n_days)
        
        # Add some persistence
        for i in range(1, len(daily_returns)):
            daily_returns[i] += 0.05 * daily_returns[i-1]
        
        perf_data = {
            'returns': daily_returns.tolist(),
            'dates': [(datetime.utcnow() - timedelta(days=i)).isoformat() 
                     for i in range(n_days, 0, -1)]
        }
        
        self.optimization_cache[cache_key] = perf_data
        return perf_data
    
    def _setup_optimization_constraints(self, 
                                      strategies: List[Dict], 
                                      additional_constraints: Optional[Dict]) -> List[Dict]:
        """Set up optimization constraints."""
        
        n_assets = len(strategies)
        constraints = []
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Additional constraints
        if additional_constraints:
            # Maximum sector exposure
            if 'max_sector_exposure' in additional_constraints:
                sector_map = self._create_sector_map(strategies)
                for sector, max_exposure in additional_constraints['max_sector_exposure'].items():
                    indices = sector_map.get(sector, [])
                    if indices:
                        constraints.append({
                            'type': 'ineq',
                            'fun': lambda w, idx=indices: max_exposure - np.sum([w[i] for i in idx])
                        })
            
            # Minimum diversification
            if 'min_positions' in additional_constraints:
                min_positions = additional_constraints['min_positions']
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w: np.sum(w > 0.01) - min_positions  # At least 1% weight
                })
        
        return constraints
    
    def _create_sector_map(self, strategies: List[Dict]) -> Dict[str, List[int]]:
        """Create mapping from sectors to strategy indices."""
        sector_map = {}
        
        for i, strategy in enumerate(strategies):
            sector = strategy.get('sector', 'general')
            if sector not in sector_map:
                sector_map[sector] = []
            sector_map[sector].append(i)
        
        return sector_map
    
    def _calculate_diversification_ratio(self, 
                                       weights: np.ndarray, 
                                       covariance_matrix: np.ndarray) -> float:
        """Calculate diversification ratio."""
        
        individual_vols = np.sqrt(np.diag(covariance_matrix))
        weighted_avg_vol = np.dot(weights, individual_vols)
        
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
        
        if portfolio_vol > 0:
            return weighted_avg_vol / portfolio_vol
        else:
            return 1.0
    
    def _calculate_concentration_index(self, weights: np.ndarray) -> float:
        """Calculate Herfindahl concentration index."""
        return np.sum(weights ** 2)
    
    def _calculate_risk_contribution(self, 
                                   weights: np.ndarray,
                                   covariance_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate risk contribution by strategy."""
        
        portfolio_var = np.dot(weights, np.dot(covariance_matrix, weights))
        
        if portfolio_var <= 0:
            return {f"strategy_{i}": 0.0 for i in range(len(weights))}
        
        marginal_contrib = np.dot(covariance_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_var
        
        return {f"strategy_{i}": risk_contrib[i] for i in range(len(weights))}
    
    def _calculate_rebalancing_urgency(self, 
                                     rebalancing_needs: Dict[str, float],
                                     market_conditions: Optional[Dict]) -> float:
        """Calculate urgency score for rebalancing."""
        
        # Base urgency from deviations
        max_deviation = max(abs(dev) for dev in rebalancing_needs.values())
        base_urgency = min(1.0, max_deviation / self.rebalance_threshold)
        
        # Adjust for market conditions
        urgency_multiplier = 1.0
        
        if market_conditions:
            volatility = market_conditions.get('volatility', 0.02)
            if volatility > 0.05:  # High volatility
                urgency_multiplier *= 1.5
            elif volatility < 0.01:  # Low volatility
                urgency_multiplier *= 0.7
        
        # Check time since last rebalance
        if self.last_rebalance:
            days_since = (datetime.utcnow() - self.last_rebalance).days
            if days_since < self.min_rebalance_interval_days:
                urgency_multiplier *= 0.3  # Reduce urgency if recently rebalanced
        
        return min(1.0, base_urgency * urgency_multiplier)
    
    def _identify_rebalancing_triggers(self, 
                                     rebalancing_needs: Dict[str, float],
                                     market_conditions: Optional[Dict]) -> List[str]:
        """Identify reasons triggering rebalancing."""
        
        triggers = []
        
        # Deviation triggers
        for strategy_id, deviation in rebalancing_needs.items():
            if abs(deviation) > self.rebalance_threshold:
                triggers.append(f"{strategy_id} deviation: {deviation:.2%}")
        
        # Market condition triggers
        if market_conditions:
            if market_conditions.get('volatility', 0) > 0.04:
                triggers.append("High market volatility")
            
            if market_conditions.get('correlation_shift', 0) > 0.2:
                triggers.append("Correlation regime shift")
        
        # Time-based triggers
        if self.last_rebalance:
            days_since = (datetime.utcnow() - self.last_rebalance).days
            if days_since > 30:  # Monthly rebalancing
                triggers.append("Scheduled monthly rebalancing")
        
        return triggers
    
    def _estimate_rebalancing_benefit(self, 
                                    current_weights: Dict[str, float],
                                    target_weights: Dict[str, float]) -> float:
        """Estimate expected benefit from rebalancing."""
        
        # Simple model: benefit proportional to deviation and expected alpha
        total_benefit = 0
        
        for strategy_id, target_weight in target_weights.items():
            current_weight = current_weights.get(strategy_id, 0)
            weight_change = target_weight - current_weight
            
            # Assume strategies with higher target weights have higher expected alpha
            expected_alpha = target_weight * 0.02  # 2% alpha assumption
            benefit = weight_change * expected_alpha
            total_benefit += benefit
        
        return total_benefit
    
    def get_current_allocation(self) -> Dict[str, Any]:
        """Get current portfolio allocation summary."""
        return {
            'current_weights': self.current_weights.copy(),
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'total_strategies': len(self.current_weights),
            'concentration_index': self._calculate_concentration_index(
                np.array(list(self.current_weights.values()))
            ) if self.current_weights else 1.0
        }