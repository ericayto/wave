"""
Strategy Generator Service

Generates trading strategies from natural language descriptions using LLM.
Validates and compiles strategies into executable format.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field, validator

from .llm_orchestrator import get_orchestrator, LLMMessage, MessageRole
from .llm_tools import get_tool_definitions
from ..models.database import get_db
from ..models.strategy import Strategy as StrategyModel


logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    """Strategy types."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    PAIRS_TRADING = "pairs_trading"
    CUSTOM = "custom"


class SignalType(str, Enum):
    """Signal types."""
    INDICATOR = "indicator"
    PRICE_ACTION = "price_action"
    VOLUME = "volume"
    FUNDAMENTAL = "fundamental"
    COMPOSITE = "composite"


@dataclass
class StrategyValidationResult:
    """Strategy validation result."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    score: float  # 0-100 quality score


class StrategyGenerator:
    """Natural language to trading strategy generator."""
    
    def __init__(self):
        self.orchestrator = get_orchestrator()
        self._supported_indicators = [
            "SMA", "EMA", "RSI", "MACD", "BOLLINGER", "STOCHASTIC",
            "ATR", "WILLIAMS_R", "CCI", "OBV", "ADX"
        ]
        self._supported_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
        self._default_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "MATIC/USDT"]
    
    async def generate_strategy(
        self,
        description: str,
        constraints: Optional[Dict[str, Any]] = None,
        user_id: int = 1
    ) -> Dict[str, Any]:
        """
        Generate trading strategy from natural language description.
        
        Args:
            description: Natural language description of desired strategy
            constraints: Additional constraints (symbols, timeframes, risk limits)
            user_id: User ID for strategy ownership
            
        Returns:
            Generated strategy definition
        """
        try:
            constraints = constraints or {}
            
            # Analyze the description to extract strategy type and requirements
            analysis = await self._analyze_description(description, constraints)
            
            # Generate the strategy using LLM
            strategy_def = await self._generate_strategy_definition(
                description, analysis, constraints
            )
            
            # Validate the generated strategy
            validation = await self._validate_strategy(strategy_def)
            
            if not validation.valid:
                # Try to fix common issues
                strategy_def = await self._fix_strategy_issues(
                    strategy_def, validation.errors
                )
                validation = await self._validate_strategy(strategy_def)
            
            # Add metadata
            strategy_def["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "user_id": user_id,
                "description": description,
                "validation_score": validation.score,
                "validation_warnings": validation.warnings,
                "auto_generated": True
            }
            
            # Store in database if valid
            if validation.valid:
                await self._store_strategy(strategy_def, user_id)
            
            return {
                "strategy": strategy_def,
                "validation": {
                    "valid": validation.valid,
                    "errors": validation.errors,
                    "warnings": validation.warnings,
                    "score": validation.score
                },
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Strategy generation failed: {e}")
            raise
    
    async def _analyze_description(
        self, 
        description: str, 
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze description to extract strategy requirements."""
        
        analysis_prompt = f"""Analyze this trading strategy description and extract key information:

"{description}"

Additional constraints: {json.dumps(constraints, indent=2)}

Return a JSON object with:
1. strategy_type: one of [trend_following, mean_reversion, breakout, momentum, arbitrage, pairs_trading, custom]
2. indicators_mentioned: list of technical indicators mentioned
3. timeframe_preference: preferred timeframe if mentioned
4. asset_preference: preferred assets/symbols if mentioned
5. risk_tolerance: low/medium/high based on description
6. entry_conditions: list of entry condition patterns
7. exit_conditions: list of exit condition patterns
8. complexity_score: 1-10 (1=simple, 10=very complex)
9. confidence: 0-100 how confident you are in the analysis

Focus on extracting concrete, actionable information."""

        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content="You are a trading strategy analyst. Extract structured information from strategy descriptions."),
            LLMMessage(role=MessageRole.USER, content=analysis_prompt)
        ]
        
        response = await self.orchestrator.generate(
            messages=messages,
            temperature=0.1,
            max_tokens=800
        )
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback analysis
            return {
                "strategy_type": "custom",
                "indicators_mentioned": [],
                "timeframe_preference": "1h",
                "asset_preference": [],
                "risk_tolerance": "medium",
                "entry_conditions": [],
                "exit_conditions": [],
                "complexity_score": 5,
                "confidence": 50
            }
    
    async def _generate_strategy_definition(
        self,
        description: str,
        analysis: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detailed strategy definition."""
        
        # Build generation prompt with examples
        generation_prompt = f"""Generate a complete trading strategy definition from this description:

"{description}"

Analysis: {json.dumps(analysis, indent=2)}
Constraints: {json.dumps(constraints, indent=2)}

Available indicators: {', '.join(self._supported_indicators)}
Supported timeframes: {', '.join(self._supported_timeframes)}

Generate a strategy following this EXACT JSON schema:

{{
  "name": "string (descriptive name)",
  "version": "1.0.0",
  "description": "string (detailed description)",
  "instrument_universe": ["KRAKEN:BTC/USDT", "KRAKEN:ETH/USDT"],
  "timeframes": ["1h"],
  "signals": [
    {{
      "id": "signal_name",
      "type": "indicator",
      "indicator": "RSI",
      "params": {{"period": 14, "threshold": 30}}
    }}
  ],
  "entries": [
    {{
      "when": "signal_name and volume_confirm",
      "action": {{"side": "buy", "size_pct": 0.25}}
    }}
  ],
  "exits": [
    {{
      "when": "take_profit(2.0) or stop_loss(1.0) or opposite_signal"
    }}
  ],
  "risk": {{
    "max_position_pct": 0.25,
    "daily_loss_limit_pct": 2.0,
    "max_orders_per_hour": 6
  }},
  "notes": "Implementation notes and rationale"
}}

Requirements:
1. Use only supported indicators and timeframes
2. Include proper entry/exit logic  
3. Set appropriate risk limits
4. Make signals realistic and testable
5. Provide clear implementation notes

Return ONLY the JSON - no other text."""

        messages = [
            LLMMessage(
                role=MessageRole.SYSTEM, 
                content="You are an expert trading strategy developer. Generate precise, executable trading strategies."
            ),
            LLMMessage(role=MessageRole.USER, content=generation_prompt)
        ]
        
        response = await self.orchestrator.generate(
            messages=messages,
            temperature=0.2,
            max_tokens=1500
        )
        
        try:
            strategy_def = json.loads(response.content)
            return strategy_def
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse strategy JSON: {e}")
            # Try to extract JSON from response
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start:end])
                except:
                    pass
            
            # Return minimal fallback strategy
            return self._get_fallback_strategy(description, analysis)
    
    def _get_fallback_strategy(
        self, 
        description: str, 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate fallback strategy when LLM generation fails."""
        
        strategy_type = analysis.get('strategy_type', 'trend_following')
        
        if strategy_type == 'trend_following':
            return {
                "name": "LLM Generated Trend Following Strategy",
                "version": "1.0.0",
                "description": description,
                "instrument_universe": ["KRAKEN:BTC/USDT"],
                "timeframes": ["1h"],
                "signals": [
                    {
                        "id": "sma_cross",
                        "type": "indicator",
                        "indicator": "SMA_CROSS",
                        "params": {"fast": 20, "slow": 50}
                    }
                ],
                "entries": [
                    {
                        "when": "sma_cross.bullish",
                        "action": {"side": "buy", "size_pct": 0.25}
                    }
                ],
                "exits": [
                    {
                        "when": "sma_cross.bearish or stop_loss(2.0) or take_profit(4.0)"
                    }
                ],
                "risk": {
                    "max_position_pct": 0.25,
                    "daily_loss_limit_pct": 2.0,
                    "max_orders_per_hour": 4
                },
                "notes": f"Fallback strategy generated from: {description}"
            }
        
        else:  # mean_reversion fallback
            return {
                "name": "LLM Generated Mean Reversion Strategy",
                "version": "1.0.0", 
                "description": description,
                "instrument_universe": ["KRAKEN:BTC/USDT"],
                "timeframes": ["15m"],
                "signals": [
                    {
                        "id": "rsi_oversold",
                        "type": "indicator",
                        "indicator": "RSI",
                        "params": {"period": 14, "threshold": 30}
                    },
                    {
                        "id": "rsi_overbought",
                        "type": "indicator",
                        "indicator": "RSI",
                        "params": {"period": 14, "threshold": 70}
                    }
                ],
                "entries": [
                    {
                        "when": "rsi_oversold",
                        "action": {"side": "buy", "size_pct": 0.2}
                    }
                ],
                "exits": [
                    {
                        "when": "rsi_overbought or stop_loss(1.5) or take_profit(2.5)"
                    }
                ],
                "risk": {
                    "max_position_pct": 0.2,
                    "daily_loss_limit_pct": 1.5,
                    "max_orders_per_hour": 6
                },
                "notes": f"Fallback strategy generated from: {description}"
            }
    
    async def _validate_strategy(self, strategy_def: Dict[str, Any]) -> StrategyValidationResult:
        """Validate generated strategy definition."""
        
        errors = []
        warnings = []
        score = 100.0
        
        # Required fields
        required_fields = [
            'name', 'version', 'instrument_universe', 'timeframes', 
            'signals', 'entries', 'exits', 'risk'
        ]
        
        for field in required_fields:
            if field not in strategy_def:
                errors.append(f"Missing required field: {field}")
                score -= 15
        
        # Validate instrument universe
        if 'instrument_universe' in strategy_def:
            symbols = strategy_def['instrument_universe']
            if not isinstance(symbols, list) or len(symbols) == 0:
                errors.append("instrument_universe must be non-empty list")
                score -= 10
            else:
                for symbol in symbols:
                    if not isinstance(symbol, str) or '/' not in symbol:
                        warnings.append(f"Invalid symbol format: {symbol}")
                        score -= 2
        
        # Validate timeframes
        if 'timeframes' in strategy_def:
            timeframes = strategy_def['timeframes']
            if not isinstance(timeframes, list) or len(timeframes) == 0:
                errors.append("timeframes must be non-empty list")
                score -= 10
            else:
                for tf in timeframes:
                    if tf not in self._supported_timeframes:
                        warnings.append(f"Unsupported timeframe: {tf}")
                        score -= 3
        
        # Validate signals
        if 'signals' in strategy_def:
            signals = strategy_def['signals']
            if not isinstance(signals, list):
                errors.append("signals must be a list")
                score -= 15
            else:
                signal_ids = set()
                for signal in signals:
                    if not isinstance(signal, dict):
                        errors.append("Each signal must be an object")
                        score -= 5
                        continue
                    
                    if 'id' not in signal:
                        errors.append("Signal missing 'id' field")
                        score -= 5
                    else:
                        if signal['id'] in signal_ids:
                            errors.append(f"Duplicate signal id: {signal['id']}")
                            score -= 10
                        signal_ids.add(signal['id'])
                    
                    if signal.get('type') == 'indicator':
                        indicator = signal.get('indicator', '')
                        if indicator not in self._supported_indicators:
                            warnings.append(f"Unsupported indicator: {indicator}")
                            score -= 2
        
        # Validate entries and exits
        for section in ['entries', 'exits']:
            if section in strategy_def:
                items = strategy_def[section]
                if not isinstance(items, list) or len(items) == 0:
                    errors.append(f"{section} must be non-empty list")
                    score -= 10
                
                for item in items:
                    if not isinstance(item, dict) or 'when' not in item:
                        errors.append(f"Invalid {section[:-1]} format")
                        score -= 5
        
        # Validate risk parameters
        if 'risk' in strategy_def:
            risk = strategy_def['risk']
            if not isinstance(risk, dict):
                errors.append("risk must be an object")
                score -= 10
            else:
                # Check reasonable risk limits
                max_pos = risk.get('max_position_pct', 0)
                if max_pos <= 0 or max_pos > 1.0:
                    warnings.append("max_position_pct should be between 0 and 1.0")
                    score -= 3
                
                daily_loss = risk.get('daily_loss_limit_pct', 0)
                if daily_loss <= 0 or daily_loss > 0.1:
                    warnings.append("daily_loss_limit_pct should be between 0 and 0.1 (10%)")
                    score -= 3
        
        # Overall quality checks
        if len(strategy_def.get('signals', [])) == 0:
            errors.append("Strategy must have at least one signal")
            score -= 20
        
        if len(strategy_def.get('entries', [])) == 0:
            errors.append("Strategy must have at least one entry condition")  
            score -= 20
        
        if len(strategy_def.get('exits', [])) == 0:
            warnings.append("Strategy should have explicit exit conditions")
            score -= 5
        
        # Ensure score is in valid range
        score = max(0.0, min(100.0, score))
        valid = len(errors) == 0 and score >= 50.0
        
        return StrategyValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            score=score
        )
    
    async def _fix_strategy_issues(
        self, 
        strategy_def: Dict[str, Any], 
        errors: List[str]
    ) -> Dict[str, Any]:
        """Attempt to fix common strategy issues."""
        
        fixed_strategy = strategy_def.copy()
        
        # Fix missing required fields
        if 'name' not in fixed_strategy:
            fixed_strategy['name'] = "Generated Strategy"
        
        if 'version' not in fixed_strategy:
            fixed_strategy['version'] = "1.0.0"
        
        if 'instrument_universe' not in fixed_strategy or not fixed_strategy['instrument_universe']:
            fixed_strategy['instrument_universe'] = ["KRAKEN:BTC/USDT"]
        
        if 'timeframes' not in fixed_strategy or not fixed_strategy['timeframes']:
            fixed_strategy['timeframes'] = ["1h"]
        
        if 'signals' not in fixed_strategy or not fixed_strategy['signals']:
            # Add basic RSI signal
            fixed_strategy['signals'] = [
                {
                    "id": "rsi_signal",
                    "type": "indicator",
                    "indicator": "RSI",
                    "params": {"period": 14, "threshold": 30}
                }
            ]
        
        if 'entries' not in fixed_strategy or not fixed_strategy['entries']:
            fixed_strategy['entries'] = [
                {
                    "when": "rsi_signal",
                    "action": {"side": "buy", "size_pct": 0.25}
                }
            ]
        
        if 'exits' not in fixed_strategy or not fixed_strategy['exits']:
            fixed_strategy['exits'] = [
                {
                    "when": "stop_loss(2.0) or take_profit(4.0)"
                }
            ]
        
        if 'risk' not in fixed_strategy or not isinstance(fixed_strategy['risk'], dict):
            fixed_strategy['risk'] = {
                "max_position_pct": 0.25,
                "daily_loss_limit_pct": 2.0,
                "max_orders_per_hour": 4
            }
        
        return fixed_strategy
    
    async def _store_strategy(self, strategy_def: Dict[str, Any], user_id: int):
        """Store validated strategy in database."""
        try:
            async with get_db() as db:
                strategy = StrategyModel(
                    user_id=user_id,
                    name=strategy_def['name'],
                    version=strategy_def['version'],
                    json=json.dumps(strategy_def),
                    status='draft'  # Start as draft until user approves
                )
                db.add(strategy)
                await db.commit()
                await db.refresh(strategy)
                
                # Add strategy_id to definition
                strategy_def['id'] = strategy.id
                
                logger.info(f"Stored strategy '{strategy_def['name']}' for user {user_id}")
                
        except Exception as e:
            logger.error(f"Failed to store strategy: {e}")
    
    async def refine_strategy(
        self,
        strategy_def: Dict[str, Any],
        refinement_request: str,
        user_id: int = 1
    ) -> Dict[str, Any]:
        """Refine an existing strategy based on user feedback."""
        
        refinement_prompt = f"""Refine this trading strategy based on the user's request:

Current Strategy:
{json.dumps(strategy_def, indent=2)}

User Request: "{refinement_request}"

Modify the strategy to address the user's request while maintaining:
1. Valid JSON format
2. All required fields
3. Reasonable risk parameters
4. Executable logic

Return the complete refined strategy JSON."""

        messages = [
            LLMMessage(
                role=MessageRole.SYSTEM,
                content="You are an expert trading strategy developer. Refine strategies based on user feedback."
            ),
            LLMMessage(role=MessageRole.USER, content=refinement_prompt)
        ]
        
        response = await self.orchestrator.generate(
            messages=messages,
            temperature=0.1,
            max_tokens=1500
        )
        
        try:
            refined_strategy = json.loads(response.content)
            
            # Validate refined strategy
            validation = await self._validate_strategy(refined_strategy)
            
            if validation.valid:
                # Update metadata
                refined_strategy.setdefault("metadata", {})
                refined_strategy["metadata"]["refined_at"] = datetime.now().isoformat()
                refined_strategy["metadata"]["refinement_request"] = refinement_request
                
                return {
                    "strategy": refined_strategy,
                    "validation": {
                        "valid": validation.valid,
                        "errors": validation.errors,
                        "warnings": validation.warnings,
                        "score": validation.score
                    }
                }
            else:
                # Return original if refinement failed validation
                return {
                    "strategy": strategy_def,
                    "validation": {
                        "valid": False,
                        "errors": validation.errors + ["Refinement failed validation"],
                        "warnings": validation.warnings,
                        "score": validation.score
                    },
                    "error": "Refined strategy failed validation"
                }
                
        except json.JSONDecodeError:
            return {
                "strategy": strategy_def,
                "error": "Failed to parse refined strategy JSON"
            }


# Global generator instance
_generator: Optional[StrategyGenerator] = None


def get_strategy_generator() -> StrategyGenerator:
    """Get global strategy generator instance."""
    global _generator
    if _generator is None:
        _generator = StrategyGenerator()
    return _generator