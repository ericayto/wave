"""
Advanced Backtesting Engine
Production-grade backtesting with walk-forward analysis, Monte Carlo simulation, and stress testing.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
import logging
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
import itertools

from ..config.settings import get_settings
from ..services.event_bus import EventBus

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Comprehensive backtest result."""
    strategy_id: str
    start_date: str
    end_date: str
    
    # Performance metrics
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    
    # Trading statistics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Risk metrics
    var_95: float
    cvar_95: float
    
    # Equity curve and trade details
    equity_curve: List[float]
    trade_history: List[Dict]
    
    # Metadata
    parameters: Dict[str, Any]
    data_quality_score: float
    
@dataclass
class WalkForwardResult:
    """Walk-forward analysis result."""
    strategy_id: str
    optimization_windows: List[Dict]
    out_of_sample_performance: List[Dict]
    
    # Stability metrics
    parameter_stability: Dict[str, float]
    performance_consistency: float
    overfitting_score: float
    
    # Aggregated results
    avg_is_return: float  # In-sample
    avg_oos_return: float  # Out-of-sample
    performance_decay: float
    
@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result."""
    strategy_id: str
    n_simulations: int
    
    # Return distributions
    return_percentiles: Dict[str, float]  # 5%, 25%, 50%, 75%, 95%
    drawdown_percentiles: Dict[str, float]
    
    # Risk metrics
    probability_of_loss: float
    expected_return: float
    expected_drawdown: float
    var_confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Stress scenarios
    worst_case_scenario: Dict[str, float]
    best_case_scenario: Dict[str, float]

@dataclass
class StressTestResult:
    """Stress testing result."""
    strategy_id: str
    scenarios_tested: List[str]
    
    # Historical stress events
    dot_com_crash: Dict[str, float]
    financial_crisis_2008: Dict[str, float]
    covid_crash_2020: Dict[str, float]
    
    # Synthetic stress scenarios
    high_volatility_shock: Dict[str, float]
    liquidity_crisis: Dict[str, float]
    correlation_breakdown: Dict[str, float]
    
    # Overall stress score
    stress_score: float  # 0-100, higher = more resilient
    vulnerabilities: List[str]

class AdvancedBacktester:
    """Production-grade backtesting with advanced analysis."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.settings = get_settings()
        
        # Execution settings
        self.max_workers = 4
        self.cache_enabled = True
        
        # Historical data cache
        self.price_data_cache = {}
        self.stress_scenarios_cache = {}
        
    async def comprehensive_backtest(self, 
                                   strategy_def: Dict, 
                                   start_date: str,
                                   end_date: str,
                                   benchmark: Optional[str] = None) -> BacktestResult:
        """Run comprehensive backtest with all analysis."""
        
        logger.info(f"Starting comprehensive backtest for strategy {strategy_def.get('name')}")
        
        # Get historical data
        price_data = await self._get_historical_data(
            strategy_def['instrument_universe'],
            start_date,
            end_date
        )
        
        # Run backtest simulation
        result = await self._run_backtest_simulation(strategy_def, price_data)
        
        # Calculate comprehensive metrics
        result = await self._calculate_comprehensive_metrics(result, benchmark)
        
        # Validate data quality
        result.data_quality_score = await self._assess_data_quality(price_data)
        
        logger.info(f"Backtest completed: {result.total_return:.2%} return, {result.sharpe_ratio:.2f} Sharpe")
        
        return result
    
    async def walk_forward_analysis(self, 
                                  strategy_def: Dict,
                                  window_size: int = 252,  # 1 year
                                  step_size: int = 21,     # 1 month
                                  optimization_target: str = 'sharpe_ratio') -> WalkForwardResult:
        """Perform walk-forward optimization analysis."""
        
        logger.info("Starting walk-forward analysis...")
        
        strategy_name = strategy_def.get('name', 'unknown')
        instrument_universe = strategy_def['instrument_universe']
        
        # Get extended historical data (need extra for walk-forward)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=window_size * 3)  # 3x window for sufficient data
        
        price_data = await self._get_historical_data(
            instrument_universe,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if len(price_data) < window_size + step_size:
            raise ValueError("Insufficient historical data for walk-forward analysis")
        
        optimization_windows = []
        oos_performance = []
        parameter_history = []
        
        # Walk-forward windows
        current_start = 0
        
        while current_start + window_size + step_size <= len(price_data):
            # In-sample optimization window
            is_start = current_start
            is_end = current_start + window_size
            
            # Out-of-sample testing window
            oos_start = is_end
            oos_end = min(oos_start + step_size, len(price_data))
            
            logger.info(f"Optimizing window {len(optimization_windows) + 1}: "
                       f"IS {is_start}-{is_end}, OOS {oos_start}-{oos_end}")
            
            # Optimize parameters on in-sample data
            is_data = price_data.iloc[is_start:is_end]
            optimal_params = await self._optimize_parameters(
                strategy_def, is_data, optimization_target
            )
            
            # Test on out-of-sample data
            oos_data = price_data.iloc[oos_start:oos_end]
            oos_result = await self._run_backtest_with_params(
                strategy_def, optimal_params, oos_data
            )
            
            optimization_windows.append({
                'window_id': len(optimization_windows) + 1,
                'is_start': is_start,
                'is_end': is_end,
                'oos_start': oos_start,
                'oos_end': oos_end,
                'optimal_params': optimal_params,
                'is_performance': await self._run_backtest_with_params(
                    strategy_def, optimal_params, is_data
                )
            })
            
            oos_performance.append(oos_result)
            parameter_history.append(optimal_params)
            
            current_start += step_size
        
        # Analyze parameter stability
        parameter_stability = self._analyze_parameter_stability(parameter_history)
        
        # Calculate performance consistency
        oos_returns = [p['total_return'] for p in oos_performance]
        performance_consistency = 1.0 - (np.std(oos_returns) / np.mean(oos_returns) if np.mean(oos_returns) != 0 else 0)
        
        # Calculate overfitting score
        is_returns = [w['is_performance']['total_return'] for w in optimization_windows]
        overfitting_score = self._calculate_overfitting_score(is_returns, oos_returns)
        
        return WalkForwardResult(
            strategy_id=strategy_name,
            optimization_windows=optimization_windows,
            out_of_sample_performance=oos_performance,
            parameter_stability=parameter_stability,
            performance_consistency=max(0.0, performance_consistency),
            overfitting_score=overfitting_score,
            avg_is_return=np.mean(is_returns),
            avg_oos_return=np.mean(oos_returns),
            performance_decay=np.mean(is_returns) - np.mean(oos_returns)
        )
    
    async def monte_carlo_simulation(self, 
                                   strategy_def: Dict,
                                   base_backtest: BacktestResult,
                                   n_simulations: int = 1000,
                                   confidence_levels: List[float] = [0.05, 0.95]) -> MonteCarloResult:
        """Run Monte Carlo simulation for robustness testing."""
        
        logger.info(f"Starting Monte Carlo simulation with {n_simulations} runs...")
        
        # Extract returns from base backtest
        base_returns = np.diff(base_backtest.equity_curve) / base_backtest.equity_curve[:-1]
        
        # Bootstrap simulation parameters
        return_mean = np.mean(base_returns)
        return_std = np.std(base_returns)
        
        # Run simulations in parallel
        simulation_results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i in range(n_simulations):
                future = executor.submit(
                    self._run_single_monte_carlo,
                    strategy_def,
                    return_mean,
                    return_std,
                    len(base_returns),
                    i
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    simulation_results.append(result)
                except Exception as e:
                    logger.warning(f"Monte Carlo simulation failed: {e}")
        
        if not simulation_results:
            raise ValueError("All Monte Carlo simulations failed")
        
        # Analyze simulation results
        final_returns = [r['final_return'] for r in simulation_results]
        max_drawdowns = [r['max_drawdown'] for r in simulation_results]
        
        # Calculate percentiles
        return_percentiles = {
            '5%': np.percentile(final_returns, 5),
            '25%': np.percentile(final_returns, 25),
            '50%': np.percentile(final_returns, 50),
            '75%': np.percentile(final_returns, 75),
            '95%': np.percentile(final_returns, 95)
        }
        
        drawdown_percentiles = {
            '5%': np.percentile(max_drawdowns, 5),
            '25%': np.percentile(max_drawdowns, 25),
            '50%': np.percentile(max_drawdowns, 50),
            '75%': np.percentile(max_drawdowns, 75),
            '95%': np.percentile(max_drawdowns, 95)
        }
        
        # Risk analysis
        probability_of_loss = sum(1 for r in final_returns if r < 0) / len(final_returns)
        
        # VaR confidence intervals
        var_95 = np.percentile(final_returns, 5)
        var_confidence = stats.bootstrap(
            (final_returns,), 
            lambda x: np.percentile(x, 5), 
            n_resamples=1000,
            confidence_level=0.95
        )
        
        return MonteCarloResult(
            strategy_id=strategy_def.get('name', 'unknown'),
            n_simulations=len(simulation_results),
            return_percentiles=return_percentiles,
            drawdown_percentiles=drawdown_percentiles,
            probability_of_loss=probability_of_loss,
            expected_return=np.mean(final_returns),
            expected_drawdown=np.mean(max_drawdowns),
            var_confidence_intervals={
                'var_95': (var_confidence.confidence_interval.low, var_confidence.confidence_interval.high)
            },
            worst_case_scenario={
                'return': min(final_returns),
                'drawdown': max(max_drawdowns),
                'scenario_id': simulation_results[np.argmin(final_returns)]['simulation_id']
            },
            best_case_scenario={
                'return': max(final_returns),
                'drawdown': min(max_drawdowns),
                'scenario_id': simulation_results[np.argmax(final_returns)]['simulation_id']
            }
        )
    
    def _run_single_monte_carlo(self, 
                               strategy_def: Dict,
                               return_mean: float,
                               return_std: float,
                               n_periods: int,
                               simulation_id: int) -> Dict:
        """Run a single Monte Carlo simulation."""
        
        # Generate synthetic returns
        np.random.seed(simulation_id)  # For reproducibility
        synthetic_returns = np.random.normal(return_mean, return_std, n_periods)
        
        # Calculate equity curve
        equity_curve = np.cumprod(1 + synthetic_returns)
        
        # Calculate metrics
        final_return = equity_curve[-1] - 1
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(min(drawdown))
        
        return {
            'simulation_id': simulation_id,
            'final_return': final_return,
            'max_drawdown': max_drawdown,
            'volatility': np.std(synthetic_returns),
            'sharpe_ratio': np.mean(synthetic_returns) / np.std(synthetic_returns) if np.std(synthetic_returns) != 0 else 0
        }
    
    async def stress_testing(self, 
                           strategy_def: Dict,
                           base_backtest: BacktestResult) -> StressTestResult:
        """Apply stress testing scenarios."""
        
        logger.info("Running comprehensive stress testing...")
        
        strategy_name = strategy_def.get('name', 'unknown')
        
        # Historical stress scenarios
        historical_results = {}
        
        # Test major historical events
        stress_scenarios = {
            'dot_com_crash': {
                'start': '2000-03-10',
                'end': '2002-10-09',
                'description': 'Dot-com bubble crash (2000-2002)'
            },
            'financial_crisis_2008': {
                'start': '2007-10-09',
                'end': '2009-03-09', 
                'description': 'Global Financial Crisis (2007-2009)'
            },
            'covid_crash_2020': {
                'start': '2020-02-19',
                'end': '2020-04-07',
                'description': 'COVID-19 market crash (Feb-Apr 2020)'
            }
        }
        
        for scenario_name, scenario_data in stress_scenarios.items():
            try:
                # Get data for stress period
                stress_data = await self._get_historical_data(
                    strategy_def['instrument_universe'],
                    scenario_data['start'],
                    scenario_data['end']
                )
                
                if len(stress_data) > 20:  # Minimum data requirement
                    stress_result = await self._run_backtest_simulation(strategy_def, stress_data)
                    historical_results[scenario_name] = {
                        'total_return': stress_result.total_return,
                        'max_drawdown': stress_result.max_drawdown,
                        'sharpe_ratio': stress_result.sharpe_ratio,
                        'volatility': stress_result.volatility
                    }
                else:
                    historical_results[scenario_name] = {'error': 'Insufficient data'}
                    
            except Exception as e:
                logger.warning(f"Failed to test {scenario_name}: {e}")
                historical_results[scenario_name] = {'error': str(e)}
        
        # Synthetic stress scenarios
        synthetic_results = await self._run_synthetic_stress_tests(strategy_def, base_backtest)
        
        # Calculate overall stress score
        stress_score = self._calculate_stress_score(historical_results, synthetic_results)
        
        # Identify vulnerabilities
        vulnerabilities = self._identify_vulnerabilities(historical_results, synthetic_results)
        
        return StressTestResult(
            strategy_id=strategy_name,
            scenarios_tested=list(stress_scenarios.keys()) + list(synthetic_results.keys()),
            dot_com_crash=historical_results.get('dot_com_crash', {}),
            financial_crisis_2008=historical_results.get('financial_crisis_2008', {}),
            covid_crash_2020=historical_results.get('covid_crash_2020', {}),
            high_volatility_shock=synthetic_results.get('high_volatility_shock', {}),
            liquidity_crisis=synthetic_results.get('liquidity_crisis', {}),
            correlation_breakdown=synthetic_results.get('correlation_breakdown', {}),
            stress_score=stress_score,
            vulnerabilities=vulnerabilities
        )
    
    async def _run_synthetic_stress_tests(self, 
                                        strategy_def: Dict,
                                        base_backtest: BacktestResult) -> Dict[str, Dict]:
        """Run synthetic stress test scenarios."""
        
        base_returns = np.diff(base_backtest.equity_curve) / base_backtest.equity_curve[:-1]
        
        synthetic_scenarios = {}
        
        # High volatility shock (3x normal volatility)
        high_vol_returns = base_returns + np.random.normal(0, np.std(base_returns) * 2, len(base_returns))
        synthetic_scenarios['high_volatility_shock'] = self._analyze_stress_returns(high_vol_returns)
        
        # Liquidity crisis (increased correlation + wider spreads)
        liquidity_returns = base_returns * 0.8 + np.random.normal(-0.001, 0.005, len(base_returns))
        synthetic_scenarios['liquidity_crisis'] = self._analyze_stress_returns(liquidity_returns)
        
        # Correlation breakdown (randomized returns)
        correlation_returns = np.random.normal(np.mean(base_returns), np.std(base_returns) * 1.5, len(base_returns))
        synthetic_scenarios['correlation_breakdown'] = self._analyze_stress_returns(correlation_returns)
        
        return synthetic_scenarios
    
    def _analyze_stress_returns(self, returns: np.ndarray) -> Dict[str, float]:
        """Analyze stress test returns."""
        equity_curve = np.cumprod(1 + returns)
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(min(drawdown))
        
        return {
            'total_return': equity_curve[-1] - 1,
            'max_drawdown': max_drawdown,
            'volatility': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0,
            'worst_day': min(returns),
            'days_underwater': sum(1 for dd in drawdown if dd < 0),
            'recovery_time': len(returns) - np.argmax(drawdown)  # Approximate
        }
    
    def _calculate_stress_score(self, 
                              historical_results: Dict,
                              synthetic_results: Dict) -> float:
        """Calculate overall stress resilience score (0-100)."""
        
        scores = []
        
        # Score historical scenarios
        for scenario, results in historical_results.items():
            if 'error' not in results:
                # Lower drawdown = higher score
                dd_score = max(0, 100 - (results['max_drawdown'] * 500))  # 20% DD = 0 points
                
                # Positive return = bonus points
                return_score = max(0, results['total_return'] * 100)
                
                scenario_score = min(100, dd_score + return_score)
                scores.append(scenario_score)
        
        # Score synthetic scenarios
        for scenario, results in synthetic_results.items():
            dd_score = max(0, 100 - (results['max_drawdown'] * 400))
            return_score = max(0, results['total_return'] * 80)
            scenario_score = min(100, dd_score + return_score)
            scores.append(scenario_score)
        
        return np.mean(scores) if scores else 50.0  # Default middle score
    
    def _identify_vulnerabilities(self, 
                                historical_results: Dict,
                                synthetic_results: Dict) -> List[str]:
        """Identify strategy vulnerabilities."""
        vulnerabilities = []
        
        # Check historical performance
        for scenario, results in historical_results.items():
            if 'error' not in results:
                if results['max_drawdown'] > 0.3:
                    vulnerabilities.append(f"High drawdown during {scenario.replace('_', ' ')}")
                if results['total_return'] < -0.2:
                    vulnerabilities.append(f"Large losses during {scenario.replace('_', ' ')}")
        
        # Check synthetic scenarios
        for scenario, results in synthetic_results.items():
            if results['max_drawdown'] > 0.25:
                vulnerabilities.append(f"Vulnerable to {scenario.replace('_', ' ')}")
            if results['recovery_time'] > len(results) * 0.8:
                vulnerabilities.append(f"Slow recovery from {scenario.replace('_', ' ')}")
        
        return vulnerabilities[:5]  # Top 5 vulnerabilities
    
    async def parameter_sensitivity_analysis(self, 
                                           strategy_def: Dict,
                                           parameter_ranges: Dict[str, List]) -> Dict[str, Any]:
        """Analyze parameter sensitivity and optimization surfaces."""
        
        logger.info("Running parameter sensitivity analysis...")
        
        # Generate parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        # Limit combinations to prevent explosion
        max_combinations = 1000
        all_combinations = list(itertools.product(*param_values))
        
        if len(all_combinations) > max_combinations:
            # Sample randomly
            combinations = np.random.choice(
                len(all_combinations), 
                max_combinations, 
                replace=False
            )
            param_combinations = [all_combinations[i] for i in combinations]
        else:
            param_combinations = all_combinations
        
        # Test each combination
        results = []
        
        for i, param_combo in enumerate(param_combinations):
            param_dict = dict(zip(param_names, param_combo))
            
            try:
                # Run backtest with these parameters
                test_result = await self._run_backtest_with_params(
                    strategy_def, 
                    param_dict, 
                    await self._get_sample_data()
                )
                
                results.append({
                    'parameters': param_dict,
                    'sharpe_ratio': test_result.get('sharpe_ratio', 0),
                    'total_return': test_result.get('total_return', 0),
                    'max_drawdown': test_result.get('max_drawdown', 1)
                })
                
            except Exception as e:
                logger.warning(f"Parameter test {i} failed: {e}")
        
        if not results:
            raise ValueError("No parameter combinations could be tested")
        
        # Analyze sensitivity
        sensitivity_analysis = self._analyze_parameter_sensitivity(results, param_names)
        
        return {
            'total_combinations_tested': len(results),
            'parameter_sensitivity': sensitivity_analysis,
            'best_parameters': max(results, key=lambda x: x['sharpe_ratio'])['parameters'],
            'worst_parameters': min(results, key=lambda x: x['sharpe_ratio'])['parameters'],
            'optimization_surface': self._create_optimization_surface(results, param_names)
        }
    
    def _analyze_parameter_sensitivity(self, results: List[Dict], param_names: List[str]) -> Dict:
        """Analyze how sensitive strategy is to parameter changes."""
        
        sensitivity = {}
        
        for param_name in param_names:
            # Group results by parameter value
            param_groups = {}
            for result in results:
                param_value = result['parameters'][param_name]
                if param_value not in param_groups:
                    param_groups[param_value] = []
                param_groups[param_value].append(result['sharpe_ratio'])
            
            # Calculate variance explained by this parameter
            if len(param_groups) > 1:
                group_means = [np.mean(group) for group in param_groups.values()]
                total_variance = np.var([r['sharpe_ratio'] for r in results])
                between_group_variance = np.var(group_means)
                
                sensitivity[param_name] = {
                    'variance_explained': between_group_variance / total_variance if total_variance > 0 else 0,
                    'range_impact': max(group_means) - min(group_means),
                    'optimal_value': list(param_groups.keys())[np.argmax(group_means)]
                }
            else:
                sensitivity[param_name] = {
                    'variance_explained': 0,
                    'range_impact': 0,
                    'optimal_value': list(param_groups.keys())[0]
                }
        
        return sensitivity
    
    def _create_optimization_surface(self, results: List[Dict], param_names: List[str]) -> Dict:
        """Create optimization surface data for visualization."""
        
        if len(param_names) == 2:
            # 2D surface
            surface_data = {
                'x_param': param_names[0],
                'y_param': param_names[1],
                'z_metric': 'sharpe_ratio',
                'points': []
            }
            
            for result in results:
                surface_data['points'].append({
                    'x': result['parameters'][param_names[0]],
                    'y': result['parameters'][param_names[1]], 
                    'z': result['sharpe_ratio']
                })
            
            return surface_data
        
        else:
            # Multi-dimensional - return correlation matrix
            param_values = []
            sharpe_values = []
            
            for result in results:
                param_values.append([result['parameters'][p] for p in param_names])
                sharpe_values.append(result['sharpe_ratio'])
            
            correlations = {}
            for i, param in enumerate(param_names):
                param_col = [pv[i] for pv in param_values]
                correlation = np.corrcoef(param_col, sharpe_values)[0, 1]
                correlations[param] = correlation if not np.isnan(correlation) else 0
            
            return {
                'type': 'correlation_matrix',
                'parameter_correlations': correlations,
                'interpretation': 'Correlation between parameter values and Sharpe ratio'
            }
    
    # Helper methods for data and simulation
    
    async def _get_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical price data."""
        # Mock implementation - would fetch real data in production
        cache_key = f"{'+'.join(symbols)}_{start_date}_{end_date}"
        
        if cache_key in self.price_data_cache:
            return self.price_data_cache[cache_key]
        
        # Generate realistic mock data
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='D')
        
        data = {}
        for symbol in symbols:
            # Generate price series with realistic characteristics
            n_days = len(dates)
            returns = np.random.normal(0.0005, 0.02, n_days)  # ~12% annual return, 32% vol
            prices = 100 * np.cumprod(1 + returns)
            data[f"{symbol}_price"] = prices
            data[f"{symbol}_return"] = returns
        
        df = pd.DataFrame(data, index=dates)
        
        if self.cache_enabled:
            self.price_data_cache[cache_key] = df
        
        return df
    
    async def _get_sample_data(self) -> pd.DataFrame:
        """Get sample data for testing."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=90)
        
        return await self._get_historical_data(
            ['BTC/USDT'], 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    
    async def _run_backtest_simulation(self, strategy_def: Dict, price_data: pd.DataFrame) -> BacktestResult:
        """Run backtest simulation."""
        # Mock backtest implementation
        start_date = price_data.index[0].strftime('%Y-%m-%d')
        end_date = price_data.index[-1].strftime('%Y-%m-%d')
        
        # Generate mock results based on strategy
        np.random.seed(hash(strategy_def.get('name', 'default')) % 2**32)
        
        n_days = len(price_data)
        daily_returns = np.random.normal(0.0008, 0.015, n_days)  # Realistic strategy returns
        equity_curve = (10000 * np.cumprod(1 + daily_returns)).tolist()
        
        # Calculate metrics
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        annual_return = (equity_curve[-1] / equity_curve[0]) ** (252 / n_days) - 1
        volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = [(eq - pk) / pk for eq, pk in zip(equity_curve, peak)]
        max_drawdown = abs(min(drawdown))
        
        # Mock trade history
        n_trades = max(1, n_days // 10)  # Trade every 10 days on average
        trade_history = []
        
        for i in range(n_trades):
            trade_date = price_data.index[i * (n_days // n_trades)]
            trade_history.append({
                'date': trade_date.strftime('%Y-%m-%d'),
                'symbol': np.random.choice(strategy_def['instrument_universe']),
                'side': np.random.choice(['buy', 'sell']),
                'quantity': np.random.uniform(0.1, 2.0),
                'price': np.random.uniform(95, 105),
                'pnl': np.random.normal(50, 200)
            })
        
        return BacktestResult(
            strategy_id=strategy_def.get('name', 'unknown'),
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sharpe_ratio * 1.1,  # Mock
            max_drawdown=max_drawdown,
            total_trades=n_trades,
            win_rate=np.random.uniform(0.45, 0.65),
            avg_win=np.random.uniform(100, 300),
            avg_loss=np.random.uniform(-250, -80),
            profit_factor=np.random.uniform(1.1, 2.2),
            var_95=np.percentile(daily_returns, 5),
            cvar_95=np.mean(daily_returns[daily_returns <= np.percentile(daily_returns, 5)]),
            equity_curve=equity_curve,
            trade_history=trade_history,
            parameters=strategy_def.get('parameters', {}),
            data_quality_score=0.95
        )
    
    async def _run_backtest_with_params(self, 
                                      strategy_def: Dict, 
                                      parameters: Dict, 
                                      price_data: pd.DataFrame) -> Dict:
        """Run backtest with specific parameters."""
        # Mock implementation
        strategy_copy = strategy_def.copy()
        strategy_copy['parameters'] = parameters
        
        result = await self._run_backtest_simulation(strategy_copy, price_data)
        
        return {
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'volatility': result.volatility
        }
    
    async def _optimize_parameters(self, 
                                 strategy_def: Dict, 
                                 price_data: pd.DataFrame,
                                 target_metric: str) -> Dict:
        """Optimize strategy parameters."""
        # Mock optimization - would use scipy.optimize or similar in production
        return {
            'sma_short': np.random.randint(5, 25),
            'sma_long': np.random.randint(30, 70),
            'rsi_period': np.random.randint(10, 20),
            'stop_loss': np.random.uniform(0.01, 0.05)
        }
    
    def _analyze_parameter_stability(self, parameter_history: List[Dict]) -> Dict[str, float]:
        """Analyze stability of optimized parameters."""
        stability = {}
        
        if not parameter_history:
            return stability
        
        # Get all parameter names
        param_names = set()
        for params in parameter_history:
            param_names.update(params.keys())
        
        for param_name in param_names:
            values = [params.get(param_name, 0) for params in parameter_history]
            
            if len(values) > 1:
                # Calculate coefficient of variation (lower = more stable)
                cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf')
                stability[param_name] = max(0, 1 - cv)  # Convert to stability score
            else:
                stability[param_name] = 1.0
        
        return stability
    
    def _calculate_overfitting_score(self, is_returns: List[float], oos_returns: List[float]) -> float:
        """Calculate overfitting score (0-1, higher = more overfitting)."""
        if not is_returns or not oos_returns:
            return 0.5
        
        is_mean = np.mean(is_returns)
        oos_mean = np.mean(oos_returns)
        
        if is_mean <= 0:
            return 0.0  # No overfitting if IS performance is poor
        
        performance_ratio = oos_mean / is_mean if is_mean > 0 else 0
        
        # Overfitting score: 1 - (OOS/IS performance ratio)
        # If OOS = IS, score = 0 (no overfitting)
        # If OOS << IS, score approaches 1 (high overfitting)
        return max(0, min(1, 1 - performance_ratio))
    
    async def _calculate_comprehensive_metrics(self, 
                                             result: BacktestResult,
                                             benchmark: Optional[str]) -> BacktestResult:
        """Calculate comprehensive metrics including benchmark comparison."""
        # In production, would calculate additional metrics like:
        # - Information ratio vs benchmark
        # - Beta and alpha
        # - Tracking error
        # - Up/down capture ratios
        
        # For now, just return the result as-is
        return result
    
    async def _assess_data_quality(self, price_data: pd.DataFrame) -> float:
        """Assess quality of historical data."""
        # Check for missing data, outliers, etc.
        missing_pct = price_data.isnull().sum().sum() / (len(price_data) * len(price_data.columns))
        
        # Simple quality score
        quality_score = max(0, 1 - missing_pct * 10)  # Penalize missing data
        
        return min(1.0, quality_score)