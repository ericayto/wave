"""
LLM Planning and Execution Loop

Central service that coordinates LLM-driven trading decisions.
Handles planning, tool execution, risk validation, and trade execution.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .llm_orchestrator import get_orchestrator, LLMMessage, MessageRole, LLMResponse
from .llm_tools import LLMTools, ToolContext, ToolError, get_tool_definitions
from .context_manager import get_context_manager, EventType
from .strategy_generator import get_strategy_generator
from .event_bus import EventBus
from .market_data import MarketDataService
from .paper_broker import PaperBroker
from .risk_engine import RiskEngine
from .strategy_engine import StrategyEngine


logger = logging.getLogger(__name__)


class PlanIntent(str, Enum):
    """Plan intent types."""
    ANALYZE = "analyze"
    ENTER = "enter"
    EXIT = "exit" 
    REBALANCE = "rebalance"
    HOLD = "hold"
    TUNE_STRATEGY = "tune_strategy"
    GENERATE_STRATEGY = "generate_strategy"


class PlanStatus(str, Enum):
    """Plan execution status."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TradingPlan:
    """Trading plan from LLM."""
    id: str
    intent: PlanIntent
    rationale: str
    constraints_checked: List[str]
    proposed_actions: List[Dict[str, Any]]
    fallback: str
    confidence: float  # 0-100
    created_at: datetime
    status: PlanStatus = PlanStatus.PENDING
    execution_results: List[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat(),
            "execution_results": self.execution_results or []
        }


class LLMPlanner:
    """LLM-powered trading planner and executor."""
    
    def __init__(
        self,
        user_id: int,
        event_bus: EventBus,
        market_data: MarketDataService,
        paper_broker: PaperBroker,
        risk_engine: RiskEngine,
        strategy_engine: StrategyEngine
    ):
        self.user_id = user_id
        self.event_bus = event_bus
        self.market_data = market_data
        self.paper_broker = paper_broker
        self.risk_engine = risk_engine
        self.strategy_engine = strategy_engine
        
        # Initialize services
        self.orchestrator = get_orchestrator()
        self.context_manager = get_context_manager(user_id, event_bus)
        self.strategy_generator = get_strategy_generator()
        
        # Tool context
        self.tool_context = ToolContext(
            user_id=user_id,
            session_id=f"session_{datetime.now().timestamp()}",
            event_bus=event_bus,
            market_data=market_data,
            paper_broker=paper_broker,
            risk_engine=risk_engine,
            strategy_engine=strategy_engine
        )
        self.tools = LLMTools(self.tool_context)
        
        # Planning state
        self.current_plan: Optional[TradingPlan] = None
        self.planning_enabled = True
        self.planning_interval = 300  # 5 minutes
        self.last_planning_time = datetime.now()
        
        # Human interaction queue
        self.human_interactions: Dict[str, Dict[str, Any]] = {}
        
        # Start planning loop
        self._planning_task = asyncio.create_task(self._planning_loop())
    
    async def _planning_loop(self):
        """Main planning loop that runs continuously."""
        logger.info(f"Started LLM planning loop for user {self.user_id}")
        
        while self.planning_enabled:
            try:
                # Check if it's time for a planning cycle
                time_since_last = (datetime.now() - self.last_planning_time).total_seconds()
                if time_since_last >= self.planning_interval:
                    await self._execute_planning_cycle()
                    self.last_planning_time = datetime.now()
                
                # Process any pending human interactions
                await self._process_human_interactions()
                
                # Sleep for a short interval
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in planning loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _execute_planning_cycle(self):
        """Execute a complete planning cycle."""
        try:
            logger.info("Executing LLM planning cycle")
            
            # Record planning cycle start
            await self.context_manager.record_event(
                EventType.PLAN,
                {"action": "planning_cycle_start"},
                {"timestamp": datetime.now().isoformat()}
            )
            
            # Build market context query
            query = await self._build_market_query()
            
            # Generate and execute plan
            plan = await self.generate_plan(query)
            
            if plan:
                execution_result = await self.execute_plan(plan)
                
                # Record planning cycle completion
                await self.context_manager.record_event(
                    EventType.PLAN,
                    {
                        "action": "planning_cycle_complete",
                        "plan": plan.to_dict(),
                        "execution_result": execution_result
                    }
                )
                
                # Emit planning event
                await self.event_bus.publish("llm.planning_cycle", {
                    "user_id": self.user_id,
                    "plan": plan.to_dict(),
                    "execution_result": execution_result,
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Planning cycle failed: {e}")
            
            # Record planning failure
            await self.context_manager.record_event(
                EventType.ERROR,
                {
                    "action": "planning_cycle_failed",
                    "error": str(e)
                }
            )
    
    async def _build_market_query(self) -> str:
        """Build contextual market query for planning."""
        
        # Get current portfolio state
        portfolio = await self.tools.get_portfolio()
        
        # Get market snapshot for main symbols
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        market_data = await self.tools.get_market_snapshot(
            symbols=symbols,
            fields=["price", "change_24h", "volume", "rsi", "sma"]
        )
        
        # Build query based on current state
        portfolio_value = portfolio.get("total_value", 0)
        unrealized_pnl = portfolio.get("unrealized_pnl", 0)
        
        query = f"""Current Portfolio Analysis and Decision Request:

Portfolio Status:
- Total Value: ${portfolio_value:,.2f}
- Unrealized P&L: ${unrealized_pnl:,.2f}
- Active Positions: {len(portfolio.get('positions', []))}

Market Overview:
{json.dumps(market_data, indent=2)}

Please analyze the current market conditions and portfolio state, then provide:
1. Market assessment and key observations
2. Risk evaluation and any concerns
3. Trading opportunities or adjustments needed
4. Specific actions to take (if any)
5. Rationale for your recommendations

Consider both technical indicators and portfolio balance when making decisions."""

        return query
    
    async def generate_plan(self, query: str) -> Optional[TradingPlan]:
        """Generate trading plan using LLM with tools."""
        try:
            logger.info("Generating trading plan")
            
            # Build context window with recent memory
            messages, context_metadata = await self.context_manager.build_context_window(
                query=query,
                include_recent=5,
                include_rag=6
            )
            
            # Get tool definitions
            tools = get_tool_definitions()
            
            # Generate plan with function calling
            response = await self.orchestrator.generate(
                messages=messages,
                tools=tools,
                temperature=0.1,
                max_tokens=1000
            )
            
            # Process function calls if any
            if response.tool_calls:
                # Execute tool calls and continue conversation
                messages.append(LLMMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls
                ))
                
                # Execute tools
                for tool_call in response.tool_calls:
                    tool_result = await self._execute_tool_call(tool_call)
                    
                    messages.append(LLMMessage(
                        role=MessageRole.TOOL,
                        content=json.dumps(tool_result, default=str),
                        tool_call_id=tool_call["id"]
                    ))
                
                # Get final response after tool execution
                final_response = await self.orchestrator.generate(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=800
                )
                
                # Parse final plan
                plan = await self._parse_plan_response(final_response.content)
                
            else:
                # Parse plan directly from response
                plan = await self._parse_plan_response(response.content)
            
            if plan:
                self.current_plan = plan
                
                # Record plan generation
                await self.context_manager.record_event(
                    EventType.PLAN,
                    {"action": "plan_generated", "plan": plan.to_dict()}
                )
                
                return plan
            
            return None
            
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            return None
    
    async def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a tool call safely."""
        try:
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            
            logger.info(f"Executing tool: {function_name} with args: {arguments}")
            
            # Get tool method
            if hasattr(self.tools, function_name):
                tool_method = getattr(self.tools, function_name)
                result = await tool_method(**arguments)
                
                # Record tool execution
                await self.context_manager.record_event(
                    EventType.ACTION,
                    {
                        "action": "tool_executed",
                        "tool": function_name,
                        "arguments": arguments,
                        "result": result
                    }
                )
                
                return result
            else:
                error_msg = f"Unknown tool: {function_name}"
                logger.error(error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    async def _parse_plan_response(self, response_content: str) -> Optional[TradingPlan]:
        """Parse LLM response into structured trading plan."""
        try:
            # Try to extract plan from response
            # Look for structured plan format first
            
            lines = response_content.strip().split('\n')
            
            # Default plan values
            intent = PlanIntent.ANALYZE
            rationale = response_content[:500] + "..." if len(response_content) > 500 else response_content
            constraints_checked = ["risk_limits", "portfolio_exposure"]
            proposed_actions = []
            fallback = "hold"
            confidence = 70.0
            
            # Try to extract structured information
            for line in lines:
                line = line.strip().lower()
                
                # Detect intent
                if any(word in line for word in ["buy", "enter", "open position"]):
                    intent = PlanIntent.ENTER
                elif any(word in line for word in ["sell", "exit", "close position"]):
                    intent = PlanIntent.EXIT
                elif any(word in line for word in ["rebalance", "adjust", "modify"]):
                    intent = PlanIntent.REBALANCE
                elif any(word in line for word in ["hold", "wait", "no action"]):
                    intent = PlanIntent.HOLD
                elif any(word in line for word in ["strategy", "generate", "create"]):
                    intent = PlanIntent.GENERATE_STRATEGY
                
                # Extract confidence if mentioned
                if "confidence" in line:
                    # Try to extract percentage
                    import re
                    match = re.search(r'(\d+)%', line)
                    if match:
                        confidence = float(match.group(1))
            
            # Create plan
            plan_id = f"plan_{datetime.now().timestamp()}"
            
            plan = TradingPlan(
                id=plan_id,
                intent=intent,
                rationale=rationale,
                constraints_checked=constraints_checked,
                proposed_actions=proposed_actions,
                fallback=fallback,
                confidence=confidence,
                created_at=datetime.now()
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to parse plan response: {e}")
            return None
    
    async def execute_plan(self, plan: TradingPlan) -> Dict[str, Any]:
        """Execute a trading plan."""
        try:
            logger.info(f"Executing plan: {plan.id} ({plan.intent.value})")
            
            plan.status = PlanStatus.EXECUTING
            execution_results = []
            
            # Record plan execution start
            await self.context_manager.record_event(
                EventType.ACTION,
                {
                    "action": "plan_execution_start",
                    "plan_id": plan.id,
                    "intent": plan.intent.value
                }
            )
            
            # Execute based on intent
            if plan.intent == PlanIntent.ANALYZE:
                result = await self._execute_analysis_plan(plan)
                execution_results.append(result)
            
            elif plan.intent in [PlanIntent.ENTER, PlanIntent.EXIT, PlanIntent.REBALANCE]:
                result = await self._execute_trading_plan(plan)
                execution_results.append(result)
            
            elif plan.intent == PlanIntent.GENERATE_STRATEGY:
                result = await self._execute_strategy_generation_plan(plan)
                execution_results.append(result)
            
            elif plan.intent == PlanIntent.HOLD:
                result = {"action": "hold", "status": "completed", "message": "No action taken"}
                execution_results.append(result)
            
            else:
                result = {"action": "unknown", "status": "failed", "message": f"Unknown intent: {plan.intent}"}
                execution_results.append(result)
            
            plan.execution_results = execution_results
            plan.status = PlanStatus.COMPLETED
            
            # Record plan execution completion
            await self.context_manager.record_event(
                EventType.ACTION,
                {
                    "action": "plan_execution_complete",
                    "plan_id": plan.id,
                    "results": execution_results
                }
            )
            
            return {
                "plan_id": plan.id,
                "status": "completed",
                "results": execution_results
            }
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            
            plan.status = PlanStatus.FAILED
            
            await self.context_manager.record_event(
                EventType.ERROR,
                {
                    "action": "plan_execution_failed",
                    "plan_id": plan.id,
                    "error": str(e)
                }
            )
            
            return {
                "plan_id": plan.id,
                "status": "failed",
                "error": str(e)
            }
    
    async def _execute_analysis_plan(self, plan: TradingPlan) -> Dict[str, Any]:
        """Execute analysis-only plan."""
        try:
            # Get portfolio and market data
            portfolio = await self.tools.get_portfolio()
            market_snapshot = await self.tools.get_market_snapshot(
                symbols=["BTC/USDT", "ETH/USDT"],
                fields=["price", "change_24h", "rsi", "volume"]
            )
            
            return {
                "action": "analysis",
                "status": "completed",
                "portfolio": portfolio,
                "market_data": market_snapshot,
                "analysis": plan.rationale
            }
            
        except Exception as e:
            return {
                "action": "analysis",
                "status": "failed",
                "error": str(e)
            }
    
    async def _execute_trading_plan(self, plan: TradingPlan) -> Dict[str, Any]:
        """Execute trading plan (enter/exit/rebalance)."""
        try:
            # For now, just log the trading intent
            # In a full implementation, this would parse the plan rationale
            # and execute specific trades
            
            logger.info(f"Trading plan execution: {plan.rationale}")
            
            # Example: if plan mentions buying BTC
            if "buy" in plan.rationale.lower() and "btc" in plan.rationale.lower():
                # This is a mock - would need proper parsing
                order_result = await self.tools.place_order(
                    symbol="BTC/USDT",
                    side="buy",
                    quantity=0.001,  # Small test amount
                    order_type="market"
                )
                
                return {
                    "action": "trade_executed",
                    "status": "completed",
                    "order": order_result
                }
            
            return {
                "action": "trade_analyzed",
                "status": "completed",
                "message": "Trading opportunity analyzed but no specific trades executed"
            }
            
        except Exception as e:
            return {
                "action": "trade_execution",
                "status": "failed",
                "error": str(e)
            }
    
    async def _execute_strategy_generation_plan(self, plan: TradingPlan) -> Dict[str, Any]:
        """Execute strategy generation plan."""
        try:
            # Extract strategy requirements from rationale
            strategy_result = await self.strategy_generator.generate_strategy(
                description=plan.rationale,
                user_id=self.user_id
            )
            
            return {
                "action": "strategy_generated",
                "status": "completed",
                "strategy": strategy_result
            }
            
        except Exception as e:
            return {
                "action": "strategy_generation",
                "status": "failed", 
                "error": str(e)
            }
    
    async def _process_human_interactions(self):
        """Process pending human interactions."""
        for interaction_id, interaction in list(self.human_interactions.items()):
            if interaction.get("status") == "pending":
                # Check if response has been provided (would be via WebSocket)
                # For now, this is a mock
                continue
    
    async def handle_user_query(self, query: str) -> Dict[str, Any]:
        """Handle direct user query/command."""
        try:
            logger.info(f"Handling user query: {query}")
            
            # Record user input
            await self.context_manager.record_event(
                EventType.USER_INPUT,
                {"query": query}
            )
            
            # Generate immediate response plan
            plan = await self.generate_plan(query)
            
            if plan:
                # Execute plan immediately for user queries
                result = await self.execute_plan(plan)
                
                return {
                    "status": "completed",
                    "plan": plan.to_dict(),
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "failed",
                    "error": "Could not generate plan from query",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"User query handling failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def set_planning_interval(self, seconds: int):
        """Set planning cycle interval."""
        self.planning_interval = max(60, seconds)  # Minimum 1 minute
        logger.info(f"Planning interval set to {self.planning_interval} seconds")
    
    def enable_planning(self, enabled: bool):
        """Enable or disable autonomous planning."""
        self.planning_enabled = enabled
        logger.info(f"Planning {'enabled' if enabled else 'disabled'}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current planner status."""
        return {
            "user_id": self.user_id,
            "planning_enabled": self.planning_enabled,
            "planning_interval": self.planning_interval,
            "last_planning_time": self.last_planning_time.isoformat(),
            "current_plan": self.current_plan.to_dict() if self.current_plan else None,
            "pending_interactions": len(self.human_interactions),
            "context_stats": self.context_manager.get_memory_stats()
        }
    
    async def shutdown(self):
        """Shutdown the planner gracefully."""
        logger.info(f"Shutting down LLM planner for user {self.user_id}")
        self.planning_enabled = False
        
        if hasattr(self, '_planning_task'):
            self._planning_task.cancel()
            try:
                await self._planning_task
            except asyncio.CancelledError:
                pass


# Global planners per user
_planners: Dict[int, LLMPlanner] = {}


def get_llm_planner(
    user_id: int,
    event_bus: EventBus,
    market_data: MarketDataService,
    paper_broker: PaperBroker,
    risk_engine: RiskEngine,
    strategy_engine: StrategyEngine
) -> LLMPlanner:
    """Get LLM planner for user."""
    global _planners
    if user_id not in _planners:
        _planners[user_id] = LLMPlanner(
            user_id=user_id,
            event_bus=event_bus,
            market_data=market_data,
            paper_broker=paper_broker,
            risk_engine=risk_engine,
            strategy_engine=strategy_engine
        )
    return _planners[user_id]