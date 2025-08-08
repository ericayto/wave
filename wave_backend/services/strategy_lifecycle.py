"""
Strategy Lifecycle Manager
Complete strategy promotion and lifecycle management from development to production.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import numpy as np
import pandas as pd
import json
from scipy import stats

from ..config.settings import get_settings
from ..services.event_bus import EventBus

logger = logging.getLogger(__name__)

class StrategyStatus(str, Enum):
    DRAFT = "draft"
    TESTING = "testing"
    APPROVED = "approved"
    LIVE = "live"
    PAUSED = "paused"
    RETIRED = "retired"
    FAILED = "failed"

class ValidationResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"

@dataclass
class StrategyMetadata:
    """Strategy metadata for lifecycle management."""
    strategy_id: str
    name: str
    version: str
    author: str
    created_at: datetime
    
    # Lifecycle status
    status: StrategyStatus
    status_changed_at: datetime
    
    # Performance tracking
    live_performance: Optional[Dict] = None
    test_performance: Optional[Dict] = None
    
    # Risk metrics
    risk_score: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    
    # Allocation information
    allocated_capital: float = 0.0
    max_allowed_capital: float = 0.0
    
    # Lifecycle metrics
    days_in_current_status: int = 0
    total_days_live: int = 0
    retirement_reason: Optional[str] = None

@dataclass
class ValidationCheck:
    """Individual validation check result."""
    check_name: str
    result: ValidationResult
    message: str
    details: Dict[str, Any]
    severity: str  # 'critical', 'high', 'medium', 'low'
    timestamp: datetime

@dataclass
class PromotionRequest:
    """Strategy promotion request."""
    strategy_id: str
    from_status: StrategyStatus
    to_status: StrategyStatus
    requested_by: str
    requested_at: datetime
    
    # Justification
    rationale: str
    test_results: Optional[Dict] = None
    
    # Approval workflow
    approved: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # Implementation
    implemented: bool = False
    implementation_notes: Optional[str] = None

@dataclass
class ABTestResult:
    """A/B testing result between strategies."""
    test_id: str
    strategy_a: str
    strategy_b: str
    
    # Test parameters
    start_date: datetime
    end_date: datetime
    allocation_split: Tuple[float, float]  # (A%, B%)
    
    # Results
    winner: Optional[str]
    confidence: float
    significance_level: float
    
    # Performance comparison
    performance_a: Dict[str, float]
    performance_b: Dict[str, float]
    
    # Statistical analysis
    p_value: float
    effect_size: float
    power: float

@dataclass
class CapitalAllocation:
    """Capital allocation for strategy."""
    strategy_id: str
    current_allocation: float
    target_allocation: float
    
    # Allocation history
    allocation_history: List[Dict]
    
    # Constraints
    min_allocation: float
    max_allocation: float
    
    # Performance-based adjustments
    performance_multiplier: float
    risk_adjustment: float

class StrategyLifecycleManager:
    """Complete strategy promotion and lifecycle management."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.settings = get_settings()
        
        # Lifecycle settings
        self.testing_period_days = 30
        self.min_trades_for_approval = 50
        self.max_strategies_live = getattr(self.settings.portfolio, 'max_strategies', 20)
        
        # Performance thresholds for promotion
        self.promotion_thresholds = {
            'min_sharpe_ratio': 0.8,
            'max_drawdown': 0.15,
            'min_win_rate': 0.45,
            'min_profit_factor': 1.2,
            'min_trades': 20
        }
        
        # Risk management
        self.max_single_strategy_allocation = getattr(self.settings.portfolio, 'max_single_strategy_weight', 0.2)
        self.total_risk_budget = 1.0
        
        # Current state
        self.strategies = {}  # strategy_id -> StrategyMetadata
        self.pending_promotions = []
        self.active_ab_tests = []
        self.capital_allocations = {}
        
        # Background tasks
        self.monitoring_task = None
        self.rebalancing_task = None
        
    async def start(self):
        """Start strategy lifecycle manager."""
        logger.info("Starting strategy lifecycle manager...")
        
        # Load existing strategies
        await self._load_existing_strategies()
        
        # Start background monitoring
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.rebalancing_task = asyncio.create_task(self._rebalancing_loop())
        
        logger.info("Strategy lifecycle manager started")
    
    async def stop(self):
        """Stop strategy lifecycle manager."""
        logger.info("Stopping strategy lifecycle manager...")
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            
        if self.rebalancing_task:
            self.rebalancing_task.cancel()
        
        logger.info("Strategy lifecycle manager stopped")
    
    async def submit_strategy(self, 
                            strategy_definition: Dict,
                            author: str = "system") -> str:
        """Submit new strategy for lifecycle management."""
        
        strategy_id = strategy_definition.get('id') or f"strategy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Submitting new strategy: {strategy_id}")
        
        # Create metadata
        metadata = StrategyMetadata(
            strategy_id=strategy_id,
            name=strategy_definition.get('name', strategy_id),
            version=strategy_definition.get('version', '1.0.0'),
            author=author,
            created_at=datetime.utcnow(),
            status=StrategyStatus.DRAFT,
            status_changed_at=datetime.utcnow()
        )
        
        # Initial validation
        validation_results = await self._validate_strategy(strategy_definition)
        
        if any(check.result == ValidationResult.FAIL for check in validation_results):
            metadata.status = StrategyStatus.FAILED
            logger.warning(f"Strategy {strategy_id} failed initial validation")
            
            # Log failed validation
            await self.event_bus.publish("strategy_validation_failed", {
                "strategy_id": strategy_id,
                "validation_results": [asdict(check) for check in validation_results]
            })
        
        self.strategies[strategy_id] = metadata
        
        # Emit event
        await self.event_bus.publish("strategy_submitted", {
            "strategy_id": strategy_id,
            "status": metadata.status,
            "author": author
        })
        
        return strategy_id
    
    async def request_promotion(self, 
                              strategy_id: str,
                              to_status: StrategyStatus,
                              rationale: str,
                              requested_by: str = "system") -> bool:
        """Request strategy promotion to next status."""
        
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        strategy = self.strategies[strategy_id]
        
        logger.info(f"Promotion requested: {strategy_id} {strategy.status} -> {to_status}")
        
        # Validate promotion path
        if not self._is_valid_promotion(strategy.status, to_status):
            logger.warning(f"Invalid promotion path: {strategy.status} -> {to_status}")
            return False
        
        # Create promotion request
        promotion_request = PromotionRequest(
            strategy_id=strategy_id,
            from_status=strategy.status,
            to_status=to_status,
            requested_by=requested_by,
            requested_at=datetime.utcnow(),
            rationale=rationale
        )
        
        # Run promotion validation
        can_promote = await self._validate_promotion(promotion_request)
        
        if can_promote:
            # Auto-approve certain promotions
            if self._should_auto_approve(strategy.status, to_status):
                await self._approve_promotion(promotion_request)
            else:
                self.pending_promotions.append(promotion_request)
                
                # Emit event for manual review
                await self.event_bus.publish("promotion_requires_review", {
                    "strategy_id": strategy_id,
                    "from_status": strategy.status,
                    "to_status": to_status,
                    "requested_by": requested_by
                })
        
        return can_promote
    
    async def strategy_promotion_pipeline(self, strategy_id: str) -> Dict[str, Any]:
        """Multi-stage strategy promotion: Draft → Testing → Approved → Live."""
        
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        strategy = self.strategies[strategy_id]
        current_status = strategy.status
        
        logger.info(f"Running promotion pipeline for {strategy_id} (current: {current_status})")
        
        pipeline_results = {
            'strategy_id': strategy_id,
            'initial_status': current_status,
            'pipeline_steps': [],
            'final_status': current_status,
            'success': False
        }
        
        # Define promotion pipeline
        pipeline_steps = []
        
        if current_status == StrategyStatus.DRAFT:
            pipeline_steps = [StrategyStatus.TESTING, StrategyStatus.APPROVED]
        elif current_status == StrategyStatus.TESTING:
            pipeline_steps = [StrategyStatus.APPROVED]
        elif current_status == StrategyStatus.APPROVED:
            pipeline_steps = [StrategyStatus.LIVE]
        
        # Execute pipeline steps
        for target_status in pipeline_steps:
            step_result = await self._execute_pipeline_step(strategy_id, target_status)
            pipeline_results['pipeline_steps'].append(step_result)
            
            if not step_result['success']:
                break
            
            strategy.status = target_status
            strategy.status_changed_at = datetime.utcnow()
            pipeline_results['final_status'] = target_status
        
        pipeline_results['success'] = pipeline_results['final_status'] != current_status
        
        return pipeline_results
    
    async def automated_validation_suite(self, strategy_definition: Dict) -> List[ValidationCheck]:
        """Comprehensive automated validation before promotion."""
        
        logger.info(f"Running automated validation suite for {strategy_definition.get('id')}")
        
        validation_checks = []
        
        # Schema validation
        schema_check = await self._validate_strategy_schema(strategy_definition)
        validation_checks.append(schema_check)
        
        # Parameter validation
        param_check = await self._validate_parameters(strategy_definition)
        validation_checks.append(param_check)
        
        # Risk validation
        risk_check = await self._validate_risk_parameters(strategy_definition)
        validation_checks.append(risk_check)
        
        # Backtesting validation
        backtest_check = await self._validate_backtest_performance(strategy_definition)
        validation_checks.append(backtest_check)
        
        # Code quality validation (if applicable)
        if 'code' in strategy_definition:
            code_check = await self._validate_code_quality(strategy_definition['code'])
            validation_checks.append(code_check)
        
        # Performance validation
        performance_check = await self._validate_expected_performance(strategy_definition)
        validation_checks.append(performance_check)
        
        # Resource usage validation
        resource_check = await self._validate_resource_usage(strategy_definition)
        validation_checks.append(resource_check)
        
        return validation_checks
    
    async def a_b_testing_framework(self, 
                                  strategy_a: str, 
                                  strategy_b: str,
                                  test_duration_days: int = 30,
                                  allocation_split: Tuple[float, float] = (0.5, 0.5)) -> ABTestResult:
        """A/B test strategies with statistical significance."""
        
        test_id = f"ab_test_{strategy_a}_{strategy_b}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting A/B test: {test_id}")
        
        if strategy_a not in self.strategies or strategy_b not in self.strategies:
            raise ValueError("Both strategies must exist for A/B testing")
        
        # Create test configuration
        ab_test = ABTestResult(
            test_id=test_id,
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=test_duration_days),
            allocation_split=allocation_split,
            winner=None,
            confidence=0.0,
            significance_level=0.05,
            performance_a={},
            performance_b={},
            p_value=1.0,
            effect_size=0.0,
            power=0.0
        )
        
        # Run A/B test simulation (in practice, would run live test)
        ab_results = await self._simulate_ab_test(ab_test)
        
        # Statistical analysis
        statistical_results = await self._analyze_ab_test_results(ab_results)
        
        ab_test.performance_a = statistical_results['performance_a']
        ab_test.performance_b = statistical_results['performance_b']
        ab_test.p_value = statistical_results['p_value']
        ab_test.effect_size = statistical_results['effect_size']
        ab_test.power = statistical_results['power']
        ab_test.confidence = 1.0 - ab_test.p_value
        
        # Determine winner
        if ab_test.p_value < ab_test.significance_level:
            if ab_test.performance_a['sharpe_ratio'] > ab_test.performance_b['sharpe_ratio']:
                ab_test.winner = strategy_a
            else:
                ab_test.winner = strategy_b
        
        # Store test result
        self.active_ab_tests.append(ab_test)
        
        # Emit results
        await self.event_bus.publish("ab_test_completed", {
            "test_id": test_id,
            "winner": ab_test.winner,
            "confidence": ab_test.confidence,
            "p_value": ab_test.p_value
        })
        
        logger.info(f"A/B test completed: {test_id}, winner: {ab_test.winner}")
        
        return ab_test
    
    async def gradual_capital_allocation(self, strategy_id: str) -> CapitalAllocation:
        """Gradually increase capital allocation based on performance."""
        
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        strategy = self.strategies[strategy_id]
        
        # Get or create capital allocation
        if strategy_id not in self.capital_allocations:
            allocation = CapitalAllocation(
                strategy_id=strategy_id,
                current_allocation=0.0,
                target_allocation=0.01,  # Start with 1%
                allocation_history=[],
                min_allocation=0.005,   # 0.5% minimum
                max_allocation=self.max_single_strategy_allocation,
                performance_multiplier=1.0,
                risk_adjustment=1.0
            )
            self.capital_allocations[strategy_id] = allocation
        else:
            allocation = self.capital_allocations[strategy_id]
        
        # Calculate performance-based adjustment
        performance_multiplier = await self._calculate_performance_multiplier(strategy_id)
        risk_adjustment = await self._calculate_risk_adjustment(strategy_id)
        
        # Update allocation based on performance
        old_target = allocation.target_allocation
        
        # Gradual increase/decrease
        if performance_multiplier > 1.1:  # Good performance
            new_target = min(
                allocation.target_allocation * 1.2,  # 20% increase
                allocation.max_allocation
            )
        elif performance_multiplier < 0.9:  # Poor performance
            new_target = max(
                allocation.target_allocation * 0.8,  # 20% decrease
                allocation.min_allocation
            )
        else:
            new_target = allocation.target_allocation
        
        # Apply risk adjustment
        new_target *= risk_adjustment
        new_target = max(allocation.min_allocation, min(allocation.max_allocation, new_target))
        
        # Update allocation
        allocation.target_allocation = new_target
        allocation.performance_multiplier = performance_multiplier
        allocation.risk_adjustment = risk_adjustment
        
        # Record allocation change
        if abs(new_target - old_target) > 0.001:  # Material change
            allocation.allocation_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'old_target': old_target,
                'new_target': new_target,
                'performance_multiplier': performance_multiplier,
                'risk_adjustment': risk_adjustment,
                'reason': 'gradual_adjustment'
            })
            
            logger.info(f"Capital allocation adjusted for {strategy_id}: {old_target:.3f} -> {new_target:.3f}")
        
        return allocation
    
    async def automatic_strategy_retirement(self) -> List[str]:
        """Automatically retire underperforming strategies."""
        
        retired_strategies = []
        
        for strategy_id, strategy in self.strategies.items():
            if strategy.status == StrategyStatus.LIVE:
                should_retire, reason = await self._should_retire_strategy(strategy_id)
                
                if should_retire:
                    await self._retire_strategy(strategy_id, reason)
                    retired_strategies.append(strategy_id)
        
        if retired_strategies:
            logger.info(f"Automatically retired strategies: {retired_strategies}")
        
        return retired_strategies
    
    # Private implementation methods
    
    async def _load_existing_strategies(self):
        """Load existing strategies from storage."""
        # Mock implementation - would load from database
        logger.info("Loading existing strategies...")
        
        # For now, just initialize empty
        self.strategies = {}
        self.pending_promotions = []
        self.active_ab_tests = []
        self.capital_allocations = {}
    
    async def _monitoring_loop(self):
        """Background monitoring of strategy performance."""
        
        while True:
            try:
                # Update strategy metrics
                await self._update_strategy_metrics()
                
                # Check for automatic promotions
                await self._check_automatic_promotions()
                
                # Monitor live strategies
                await self._monitor_live_strategies()
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _rebalancing_loop(self):
        """Background capital rebalancing."""
        
        while True:
            try:
                # Rebalance capital allocations
                await self._rebalance_capital_allocations()
                
                # Check for retirement candidates
                await self.automatic_strategy_retirement()
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rebalancing loop: {e}")
                await asyncio.sleep(600)
    
    async def _validate_strategy(self, strategy_definition: Dict) -> List[ValidationCheck]:
        """Validate strategy definition."""
        
        checks = []
        
        # Basic schema validation
        required_fields = ['name', 'instrument_universe', 'signals']
        
        for field in required_fields:
            if field not in strategy_definition:
                checks.append(ValidationCheck(
                    check_name=f"required_field_{field}",
                    result=ValidationResult.FAIL,
                    message=f"Missing required field: {field}",
                    details={'field': field},
                    severity='critical',
                    timestamp=datetime.utcnow()
                ))
            else:
                checks.append(ValidationCheck(
                    check_name=f"required_field_{field}",
                    result=ValidationResult.PASS,
                    message=f"Required field present: {field}",
                    details={'field': field},
                    severity='low',
                    timestamp=datetime.utcnow()
                ))
        
        return checks
    
    def _is_valid_promotion(self, from_status: StrategyStatus, to_status: StrategyStatus) -> bool:
        """Check if promotion path is valid."""
        
        valid_transitions = {
            StrategyStatus.DRAFT: [StrategyStatus.TESTING, StrategyStatus.FAILED],
            StrategyStatus.TESTING: [StrategyStatus.APPROVED, StrategyStatus.FAILED, StrategyStatus.DRAFT],
            StrategyStatus.APPROVED: [StrategyStatus.LIVE, StrategyStatus.TESTING],
            StrategyStatus.LIVE: [StrategyStatus.PAUSED, StrategyStatus.RETIRED],
            StrategyStatus.PAUSED: [StrategyStatus.LIVE, StrategyStatus.RETIRED],
            StrategyStatus.FAILED: [StrategyStatus.DRAFT],
            StrategyStatus.RETIRED: []
        }
        
        return to_status in valid_transitions.get(from_status, [])
    
    async def _validate_promotion(self, promotion_request: PromotionRequest) -> bool:
        """Validate promotion request."""
        
        strategy_id = promotion_request.strategy_id
        to_status = promotion_request.to_status
        
        # Check capacity limits
        if to_status == StrategyStatus.LIVE:
            live_strategies = [s for s in self.strategies.values() if s.status == StrategyStatus.LIVE]
            
            if len(live_strategies) >= self.max_strategies_live:
                logger.warning(f"Cannot promote {strategy_id}: live strategy limit reached")
                return False
        
        # Check performance thresholds
        if to_status in [StrategyStatus.APPROVED, StrategyStatus.LIVE]:
            performance_valid = await self._check_performance_thresholds(strategy_id)
            
            if not performance_valid:
                logger.warning(f"Cannot promote {strategy_id}: performance thresholds not met")
                return False
        
        return True
    
    def _should_auto_approve(self, from_status: StrategyStatus, to_status: StrategyStatus) -> bool:
        """Check if promotion should be auto-approved."""
        
        # Auto-approve certain transitions
        auto_approve_transitions = [
            (StrategyStatus.DRAFT, StrategyStatus.TESTING),
            (StrategyStatus.TESTING, StrategyStatus.FAILED),
            (StrategyStatus.LIVE, StrategyStatus.PAUSED)
        ]
        
        return (from_status, to_status) in auto_approve_transitions
    
    async def _approve_promotion(self, promotion_request: PromotionRequest):
        """Approve and implement promotion."""
        
        strategy_id = promotion_request.strategy_id
        to_status = promotion_request.to_status
        
        promotion_request.approved = True
        promotion_request.approved_by = "system"
        promotion_request.approved_at = datetime.utcnow()
        
        # Update strategy status
        strategy = self.strategies[strategy_id]
        old_status = strategy.status
        strategy.status = to_status
        strategy.status_changed_at = datetime.utcnow()
        
        promotion_request.implemented = True
        
        # Emit event
        await self.event_bus.publish("strategy_promoted", {
            "strategy_id": strategy_id,
            "from_status": old_status,
            "to_status": to_status,
            "approved_by": "system"
        })
        
        logger.info(f"Strategy promoted: {strategy_id} {old_status} -> {to_status}")
    
    async def _execute_pipeline_step(self, strategy_id: str, target_status: StrategyStatus) -> Dict[str, Any]:
        """Execute single pipeline step."""
        
        step_result = {
            'target_status': target_status,
            'success': False,
            'checks_passed': 0,
            'checks_failed': 0,
            'validation_results': []
        }
        
        # Run validation for this step
        if target_status == StrategyStatus.TESTING:
            # Basic validation for testing
            validation_results = await self._validate_strategy({'id': strategy_id, 'name': strategy_id})
            
        elif target_status == StrategyStatus.APPROVED:
            # Performance validation for approval
            validation_results = await self._validate_performance_for_approval(strategy_id)
            
        elif target_status == StrategyStatus.LIVE:
            # Final validation for live deployment
            validation_results = await self._validate_for_live_deployment(strategy_id)
        
        else:
            validation_results = []
        
        # Count results
        for check in validation_results:
            if check.result == ValidationResult.PASS:
                step_result['checks_passed'] += 1
            else:
                step_result['checks_failed'] += 1
        
        step_result['validation_results'] = [asdict(check) for check in validation_results]
        
        # Determine if step passed
        critical_failures = [c for c in validation_results if c.severity == 'critical' and c.result == ValidationResult.FAIL]
        step_result['success'] = len(critical_failures) == 0
        
        return step_result
    
    async def _validate_strategy_schema(self, strategy_definition: Dict) -> ValidationCheck:
        """Validate strategy schema."""
        
        required_fields = ['name', 'instrument_universe', 'signals']
        missing_fields = [field for field in required_fields if field not in strategy_definition]
        
        if missing_fields:
            return ValidationCheck(
                check_name="schema_validation",
                result=ValidationResult.FAIL,
                message=f"Missing required fields: {missing_fields}",
                details={'missing_fields': missing_fields},
                severity='critical',
                timestamp=datetime.utcnow()
            )
        else:
            return ValidationCheck(
                check_name="schema_validation",
                result=ValidationResult.PASS,
                message="All required fields present",
                details={'validated_fields': required_fields},
                severity='low',
                timestamp=datetime.utcnow()
            )
    
    async def _validate_parameters(self, strategy_definition: Dict) -> ValidationCheck:
        """Validate strategy parameters."""
        
        # Mock parameter validation
        return ValidationCheck(
            check_name="parameter_validation",
            result=ValidationResult.PASS,
            message="Parameters within acceptable ranges",
            details={'parameter_count': len(strategy_definition.get('parameters', {}))},
            severity='medium',
            timestamp=datetime.utcnow()
        )
    
    async def _validate_risk_parameters(self, strategy_definition: Dict) -> ValidationCheck:
        """Validate risk parameters."""
        
        risk_params = strategy_definition.get('risk', {})
        
        # Check for basic risk controls
        required_risk_params = ['max_position_pct', 'daily_loss_limit_pct']
        missing_params = [p for p in required_risk_params if p not in risk_params]
        
        if missing_params:
            return ValidationCheck(
                check_name="risk_validation",
                result=ValidationResult.FAIL,
                message=f"Missing risk parameters: {missing_params}",
                details={'missing_risk_params': missing_params},
                severity='critical',
                timestamp=datetime.utcnow()
            )
        else:
            return ValidationCheck(
                check_name="risk_validation",
                result=ValidationResult.PASS,
                message="Risk parameters properly configured",
                details={'risk_params': list(risk_params.keys())},
                severity='medium',
                timestamp=datetime.utcnow()
            )
    
    async def _validate_backtest_performance(self, strategy_definition: Dict) -> ValidationCheck:
        """Validate backtest performance."""
        
        # Mock backtest validation
        mock_sharpe = np.random.uniform(0.5, 2.0)
        
        if mock_sharpe >= self.promotion_thresholds['min_sharpe_ratio']:
            return ValidationCheck(
                check_name="backtest_performance",
                result=ValidationResult.PASS,
                message=f"Backtest Sharpe ratio acceptable: {mock_sharpe:.2f}",
                details={'sharpe_ratio': mock_sharpe},
                severity='high',
                timestamp=datetime.utcnow()
            )
        else:
            return ValidationCheck(
                check_name="backtest_performance",
                result=ValidationResult.FAIL,
                message=f"Backtest Sharpe ratio too low: {mock_sharpe:.2f}",
                details={'sharpe_ratio': mock_sharpe, 'threshold': self.promotion_thresholds['min_sharpe_ratio']},
                severity='critical',
                timestamp=datetime.utcnow()
            )
    
    async def _validate_code_quality(self, code: str) -> ValidationCheck:
        """Validate code quality."""
        
        # Mock code quality validation
        return ValidationCheck(
            check_name="code_quality",
            result=ValidationResult.PASS,
            message="Code quality checks passed",
            details={'lines_of_code': len(code.split('\n')) if code else 0},
            severity='medium',
            timestamp=datetime.utcnow()
        )
    
    async def _validate_expected_performance(self, strategy_definition: Dict) -> ValidationCheck:
        """Validate expected performance metrics."""
        
        # Mock performance validation
        return ValidationCheck(
            check_name="expected_performance",
            result=ValidationResult.PASS,
            message="Expected performance within acceptable ranges",
            details={'validation_method': 'monte_carlo'},
            severity='high',
            timestamp=datetime.utcnow()
        )
    
    async def _validate_resource_usage(self, strategy_definition: Dict) -> ValidationCheck:
        """Validate resource usage requirements."""
        
        # Mock resource validation
        return ValidationCheck(
            check_name="resource_usage",
            result=ValidationResult.PASS,
            message="Resource requirements acceptable",
            details={'estimated_cpu': 'low', 'estimated_memory': 'low'},
            severity='low',
            timestamp=datetime.utcnow()
        )
    
    async def _simulate_ab_test(self, ab_test: ABTestResult) -> Dict:
        """Simulate A/B test results."""
        
        # Generate mock performance data for both strategies
        np.random.seed(hash(ab_test.test_id) % 2**32)
        
        # Strategy A performance
        returns_a = np.random.normal(0.0008, 0.02, 30)  # 30 days of returns
        
        # Strategy B performance (slightly different)
        returns_b = np.random.normal(0.0012, 0.018, 30)  # Slightly better
        
        return {
            'returns_a': returns_a,
            'returns_b': returns_b,
            'test_duration': 30
        }
    
    async def _analyze_ab_test_results(self, ab_results: Dict) -> Dict:
        """Analyze A/B test results statistically."""
        
        returns_a = ab_results['returns_a']
        returns_b = ab_results['returns_b']
        
        # Calculate performance metrics
        perf_a = {
            'total_return': np.prod(1 + returns_a) - 1,
            'sharpe_ratio': np.mean(returns_a) / np.std(returns_a) * np.sqrt(252) if np.std(returns_a) > 0 else 0,
            'volatility': np.std(returns_a) * np.sqrt(252),
            'max_drawdown': 0.05  # Mock
        }
        
        perf_b = {
            'total_return': np.prod(1 + returns_b) - 1,
            'sharpe_ratio': np.mean(returns_b) / np.std(returns_b) * np.sqrt(252) if np.std(returns_b) > 0 else 0,
            'volatility': np.std(returns_b) * np.sqrt(252),
            'max_drawdown': 0.04  # Mock
        }
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_ind(returns_a, returns_b)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(returns_a)**2 + np.std(returns_b)**2) / 2)
        effect_size = (np.mean(returns_b) - np.mean(returns_a)) / pooled_std if pooled_std > 0 else 0
        
        # Statistical power (simplified)
        power = 0.8 if abs(effect_size) > 0.5 else 0.6
        
        return {
            'performance_a': perf_a,
            'performance_b': perf_b,
            'p_value': abs(p_value),  # Two-tailed
            'effect_size': effect_size,
            'power': power
        }
    
    async def _calculate_performance_multiplier(self, strategy_id: str) -> float:
        """Calculate performance-based multiplier."""
        
        # Mock performance calculation
        # In practice, would analyze recent strategy performance
        
        mock_sharpe = np.random.uniform(0.5, 2.0)
        mock_returns = np.random.uniform(-0.1, 0.3)
        
        # Performance multiplier based on Sharpe and returns
        if mock_sharpe > 1.5 and mock_returns > 0.1:
            return 1.3  # Excellent performance
        elif mock_sharpe > 1.0 and mock_returns > 0.05:
            return 1.1  # Good performance
        elif mock_sharpe < 0.5 or mock_returns < -0.05:
            return 0.7  # Poor performance
        else:
            return 1.0  # Average performance
    
    async def _calculate_risk_adjustment(self, strategy_id: str) -> float:
        """Calculate risk-based adjustment."""
        
        strategy = self.strategies[strategy_id]
        
        # Risk adjustment based on drawdown and volatility
        if strategy.max_drawdown > 0.15:
            return 0.8  # High drawdown
        elif strategy.max_drawdown < 0.05:
            return 1.1  # Low drawdown
        else:
            return 1.0  # Normal drawdown
    
    async def _should_retire_strategy(self, strategy_id: str) -> Tuple[bool, str]:
        """Check if strategy should be retired."""
        
        strategy = self.strategies[strategy_id]
        
        # Check various retirement criteria
        
        # 1. Poor performance for extended period
        if (strategy.live_performance and 
            strategy.live_performance.get('sharpe_ratio', 1.0) < 0.3 and
            strategy.total_days_live > 90):
            return True, "Consistently poor performance"
        
        # 2. Excessive drawdown
        if strategy.max_drawdown > 0.25:
            return True, "Excessive drawdown"
        
        # 3. Model degradation (mock)
        if np.random.random() < 0.01:  # 1% chance
            return True, "Model degradation detected"
        
        # 4. Risk limit violations
        if strategy.risk_score > 90:
            return True, "Repeated risk limit violations"
        
        return False, ""
    
    async def _retire_strategy(self, strategy_id: str, reason: str):
        """Retire a strategy."""
        
        strategy = self.strategies[strategy_id]
        old_status = strategy.status
        
        strategy.status = StrategyStatus.RETIRED
        strategy.status_changed_at = datetime.utcnow()
        strategy.retirement_reason = reason
        
        # Zero out capital allocation
        if strategy_id in self.capital_allocations:
            self.capital_allocations[strategy_id].target_allocation = 0.0
            self.capital_allocations[strategy_id].current_allocation = 0.0
        
        # Emit event
        await self.event_bus.publish("strategy_retired", {
            "strategy_id": strategy_id,
            "reason": reason,
            "previous_status": old_status
        })
        
        logger.info(f"Strategy retired: {strategy_id} - {reason}")
    
    async def _update_strategy_metrics(self):
        """Update metrics for all strategies."""
        
        for strategy_id, strategy in self.strategies.items():
            # Update days in current status
            strategy.days_in_current_status = (datetime.utcnow() - strategy.status_changed_at).days
            
            # Update total days live
            if strategy.status == StrategyStatus.LIVE:
                strategy.total_days_live += 1
            
            # Mock performance updates
            if strategy.status == StrategyStatus.LIVE:
                strategy.live_performance = {
                    'sharpe_ratio': np.random.uniform(0.5, 2.0),
                    'total_return': np.random.uniform(-0.1, 0.3),
                    'max_drawdown': np.random.uniform(0.02, 0.20)
                }
                strategy.max_drawdown = strategy.live_performance['max_drawdown']
                strategy.risk_score = np.random.uniform(10, 80)
    
    async def _check_automatic_promotions(self):
        """Check for strategies ready for automatic promotion."""
        
        for strategy_id, strategy in self.strategies.items():
            
            # Auto-promote from testing to approved if criteria met
            if (strategy.status == StrategyStatus.TESTING and 
                strategy.days_in_current_status >= self.testing_period_days):
                
                # Check if performance criteria met
                performance_ok = await self._check_performance_thresholds(strategy_id)
                
                if performance_ok:
                    await self.request_promotion(
                        strategy_id, 
                        StrategyStatus.APPROVED, 
                        "Automatic promotion after successful testing period"
                    )
    
    async def _monitor_live_strategies(self):
        """Monitor live strategies for issues."""
        
        for strategy_id, strategy in self.strategies.items():
            if strategy.status == StrategyStatus.LIVE:
                
                # Check for performance degradation
                if (strategy.live_performance and 
                    strategy.live_performance.get('sharpe_ratio', 1.0) < 0.2):
                    
                    # Pause strategy
                    strategy.status = StrategyStatus.PAUSED
                    strategy.status_changed_at = datetime.utcnow()
                    
                    await self.event_bus.publish("strategy_auto_paused", {
                        "strategy_id": strategy_id,
                        "reason": "Performance degradation"
                    })
    
    async def _rebalance_capital_allocations(self):
        """Rebalance capital allocations across strategies."""
        
        live_strategies = [s for s in self.strategies.values() if s.status == StrategyStatus.LIVE]
        
        if not live_strategies:
            return
        
        # Update allocations based on performance
        for strategy in live_strategies:
            await self.gradual_capital_allocation(strategy.strategy_id)
        
        # Ensure total allocation doesn't exceed 100%
        total_allocation = sum(
            self.capital_allocations.get(s.strategy_id, CapitalAllocation(s.strategy_id, 0, 0, [], 0, 0, 1, 1)).target_allocation
            for s in live_strategies
        )
        
        if total_allocation > 1.0:
            # Scale down proportionally
            scale_factor = 0.95 / total_allocation  # Leave 5% buffer
            
            for strategy in live_strategies:
                if strategy.strategy_id in self.capital_allocations:
                    allocation = self.capital_allocations[strategy.strategy_id]
                    allocation.target_allocation *= scale_factor
    
    async def _check_performance_thresholds(self, strategy_id: str) -> bool:
        """Check if strategy meets performance thresholds."""
        
        # Mock performance check
        mock_metrics = {
            'sharpe_ratio': np.random.uniform(0.5, 2.0),
            'win_rate': np.random.uniform(0.4, 0.7),
            'profit_factor': np.random.uniform(1.0, 2.5),
            'max_drawdown': np.random.uniform(0.02, 0.20),
            'trade_count': np.random.randint(10, 100)
        }
        
        # Check thresholds
        checks = [
            mock_metrics['sharpe_ratio'] >= self.promotion_thresholds['min_sharpe_ratio'],
            mock_metrics['win_rate'] >= self.promotion_thresholds['min_win_rate'],
            mock_metrics['profit_factor'] >= self.promotion_thresholds['min_profit_factor'],
            mock_metrics['max_drawdown'] <= self.promotion_thresholds['max_drawdown'],
            mock_metrics['trade_count'] >= self.promotion_thresholds['min_trades']
        ]
        
        return all(checks)
    
    async def _validate_performance_for_approval(self, strategy_id: str) -> List[ValidationCheck]:
        """Validate performance for approval stage."""
        
        checks = []
        
        # Mock performance validation
        mock_sharpe = np.random.uniform(0.5, 2.0)
        
        if mock_sharpe >= self.promotion_thresholds['min_sharpe_ratio']:
            checks.append(ValidationCheck(
                check_name="approval_performance",
                result=ValidationResult.PASS,
                message="Performance meets approval criteria",
                details={'sharpe_ratio': mock_sharpe},
                severity='high',
                timestamp=datetime.utcnow()
            ))
        else:
            checks.append(ValidationCheck(
                check_name="approval_performance",
                result=ValidationResult.FAIL,
                message="Performance below approval threshold",
                details={'sharpe_ratio': mock_sharpe},
                severity='critical',
                timestamp=datetime.utcnow()
            ))
        
        return checks
    
    async def _validate_for_live_deployment(self, strategy_id: str) -> List[ValidationCheck]:
        """Validate strategy for live deployment."""
        
        checks = []
        
        # Resource validation
        checks.append(ValidationCheck(
            check_name="live_resource_check",
            result=ValidationResult.PASS,
            message="Resources available for live deployment",
            details={'cpu_available': True, 'memory_available': True},
            severity='high',
            timestamp=datetime.utcnow()
        ))
        
        # Risk validation
        checks.append(ValidationCheck(
            check_name="live_risk_check",
            result=ValidationResult.PASS,
            message="Risk parameters acceptable for live trading",
            details={'max_position_ok': True, 'stop_loss_ok': True},
            severity='critical',
            timestamp=datetime.utcnow()
        ))
        
        return checks
    
    def get_lifecycle_status(self) -> Dict[str, Any]:
        """Get current lifecycle manager status."""
        
        status_counts = {}
        for status in StrategyStatus:
            status_counts[status.value] = sum(1 for s in self.strategies.values() if s.status == status)
        
        return {
            'total_strategies': len(self.strategies),
            'status_distribution': status_counts,
            'pending_promotions': len(self.pending_promotions),
            'active_ab_tests': len(self.active_ab_tests),
            'capital_allocations': len(self.capital_allocations),
            'max_live_strategies': self.max_strategies_live
        }