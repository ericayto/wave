"""
LLM Orchestrator Service

Central service for managing LLM interactions across multiple providers.
Handles provider selection, token budgets, context management, and function calling.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict

import httpx
import tiktoken
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ..config.settings import settings
from .event_bus import EventBus


logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    AZURE = "azure" 
    OPENROUTER = "openrouter"
    LOCAL = "local"


class MessageRole(str, Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user" 
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class LLMMessage:
    """Standard message format for LLM conversations."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


@dataclass
class LLMUsageStats:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class LLMResponse:
    """Standard LLM response format."""
    content: Optional[str]
    tool_calls: Optional[List[Dict]]
    usage: LLMUsageStats
    model: str
    provider: LLMProvider
    cached: bool = False


class TokenBudgetManager:
    """Manages token budgets and cost controls."""
    
    def __init__(self):
        self.hourly_usage: Dict[str, int] = {}
        self.daily_usage: Dict[str, int] = {}
        self.last_reset_hour = datetime.now().hour
        self.last_reset_day = datetime.now().date()
    
    def check_budget(self, provider: LLMProvider, estimated_tokens: int) -> bool:
        """Check if request fits within budget."""
        self._reset_counters()
        
        config = settings.llm.providers.get(provider.value, {})
        hourly_limit = config.get('hourly_token_budget', 50000)
        daily_limit = config.get('daily_token_budget', 500000)
        
        current_hour = self.hourly_usage.get(provider.value, 0)
        current_day = self.daily_usage.get(provider.value, 0)
        
        return (current_hour + estimated_tokens <= hourly_limit and 
                current_day + estimated_tokens <= daily_limit)
    
    def record_usage(self, provider: LLMProvider, tokens: int):
        """Record token usage for budget tracking."""
        self._reset_counters()
        self.hourly_usage[provider.value] = self.hourly_usage.get(provider.value, 0) + tokens
        self.daily_usage[provider.value] = self.daily_usage.get(provider.value, 0) + tokens
    
    def _reset_counters(self):
        """Reset counters if time period has elapsed."""
        now = datetime.now()
        if now.hour != self.last_reset_hour:
            self.hourly_usage.clear()
            self.last_reset_hour = now.hour
        
        if now.date() != self.last_reset_day:
            self.daily_usage.clear()
            self.last_reset_day = now.date()


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.encoding = tiktoken.encoding_for_model("gpt-4")  # Default tokenizer
    
    @abstractmethod
    async def generate(
        self, 
        messages: List[LLMMessage], 
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response from LLM."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback: rough estimate
            return len(text.split()) * 1.3
    
    def estimate_cost(self, usage: LLMUsageStats, model: str) -> float:
        """Estimate cost in USD."""
        # Default pricing (OpenAI GPT-4o-mini rates)
        input_cost_per_1k = 0.000150
        output_cost_per_1k = 0.000600
        
        input_cost = (usage.prompt_tokens / 1000) * input_cost_per_1k
        output_cost = (usage.completion_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""
    
    def __init__(self):
        super().__init__(LLMProvider.OPENAI)
        config = settings.llm.providers.get('openai', {})
        self.client = AsyncOpenAI(
            api_key=config.get('api_key'),
            base_url=config.get('base_url', 'https://api.openai.com/v1')
        )
        self.model = config.get('model', 'gpt-4o-mini')
    
    async def generate(
        self, 
        messages: List[LLMMessage], 
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response from OpenAI."""
        try:
            # Convert messages to OpenAI format
            openai_messages = []
            for msg in messages:
                openai_msg = {"role": msg.role.value, "content": msg.content}
                if msg.name:
                    openai_msg["name"] = msg.name
                if msg.tool_calls:
                    openai_msg["tool_calls"] = msg.tool_calls
                if msg.tool_call_id:
                    openai_msg["tool_call_id"] = msg.tool_call_id
                openai_messages.append(openai_msg)
            
            # Build request parameters
            params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": kwargs.get('temperature', 0.1),
                "max_tokens": kwargs.get('max_tokens', 1000)
            }
            
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"
            
            # Make API call
            response = await self.client.chat.completions.create(**params)
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content
            tool_calls = None
            
            if choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments
                        }
                    }
                    for call in choice.message.tool_calls
                ]
            
            # Build usage stats
            usage = LLMUsageStats(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
            usage.cost_usd = self.estimate_cost(usage, self.model)
            
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                model=self.model,
                provider=self.provider
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI provider."""
    
    def __init__(self):
        super().__init__(LLMProvider.AZURE)
        config = settings.llm.providers.get('azure', {})
        self.client = AsyncOpenAI(
            api_key=config.get('api_key'),
            azure_endpoint=config.get('endpoint'),
            api_version="2024-02-15-preview"
        )
        self.model = config.get('model', 'gpt-4o-mini')
    
    async def generate(
        self, 
        messages: List[LLMMessage], 
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response from Azure OpenAI."""
        # Implementation similar to OpenAI but with Azure-specific configuration
        return await OpenAIProvider.generate(self, messages, tools, **kwargs)


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider."""
    
    def __init__(self):
        super().__init__(LLMProvider.OPENROUTER)
        config = settings.llm.providers.get('openrouter', {})
        self.client = AsyncOpenAI(
            api_key=config.get('api_key'),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = config.get('model', 'meta-llama/llama-3.1-8b-instruct:free')
    
    async def generate(
        self, 
        messages: List[LLMMessage], 
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response from OpenRouter."""
        # Implementation similar to OpenAI
        return await OpenAIProvider.generate(self, messages, tools, **kwargs)


class LocalProvider(BaseLLMProvider):
    """Local model provider (Ollama)."""
    
    def __init__(self):
        super().__init__(LLMProvider.LOCAL)
        config = settings.llm.providers.get('local', {})
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model = config.get('model', 'llama3.1:8b')
    
    async def generate(
        self, 
        messages: List[LLMMessage], 
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response from local model."""
        try:
            # Convert messages to Ollama format
            prompt = self._build_prompt(messages, tools)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                
                result = response.json()
                content = result.get('response', '')
                
                # Estimate tokens (local models don't report usage)
                prompt_tokens = self.count_tokens(prompt)
                completion_tokens = self.count_tokens(content)
                
                usage = LLMUsageStats(
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                    total_tokens=int(prompt_tokens + completion_tokens),
                    cost_usd=0.0  # Local models are free
                )
                
                return LLMResponse(
                    content=content,
                    tool_calls=None,  # Most local models don't support function calling
                    usage=usage,
                    model=self.model,
                    provider=self.provider
                )
                
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            raise
    
    def _build_prompt(self, messages: List[LLMMessage], tools: Optional[List[Dict]]) -> str:
        """Build prompt for local model."""
        prompt_parts = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                prompt_parts.append(f"<system>\n{msg.content}\n</system>")
            elif msg.role == MessageRole.USER:
                prompt_parts.append(f"<user>\n{msg.content}\n</user>")
            elif msg.role == MessageRole.ASSISTANT:
                prompt_parts.append(f"<assistant>\n{msg.content}\n</assistant>")
        
        if tools:
            tools_str = json.dumps(tools, indent=2)
            prompt_parts.insert(1, f"<tools>\n{tools_str}\n</tools>")
        
        return "\n\n".join(prompt_parts)


class LLMOrchestrator:
    """Central LLM orchestration service."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.budget_manager = TokenBudgetManager()
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.response_cache: Dict[str, LLMResponse] = {}
        self.cache_ttl = timedelta(minutes=15)
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured providers."""
        try:
            if settings.llm.providers.get('openai', {}).get('api_key'):
                self.providers[LLMProvider.OPENAI] = OpenAIProvider()
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI provider: {e}")
        
        try:
            if settings.llm.providers.get('azure', {}).get('api_key'):
                self.providers[LLMProvider.AZURE] = AzureOpenAIProvider()
        except Exception as e:
            logger.warning(f"Failed to initialize Azure provider: {e}")
        
        try:
            if settings.llm.providers.get('openrouter', {}).get('api_key'):
                self.providers[LLMProvider.OPENROUTER] = OpenRouterProvider()
        except Exception as e:
            logger.warning(f"Failed to initialize OpenRouter provider: {e}")
        
        try:
            # Local provider doesn't require API key
            self.providers[LLMProvider.LOCAL] = LocalProvider()
        except Exception as e:
            logger.warning(f"Failed to initialize local provider: {e}")
        
        if not self.providers:
            logger.error("No LLM providers available!")
    
    async def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict]] = None,
        provider: Optional[LLMProvider] = None,
        use_cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """Generate response using best available provider."""
        
        # Select provider
        if provider and provider in self.providers:
            selected_provider = provider
        else:
            selected_provider = self._select_best_provider(messages, tools)
        
        if not selected_provider:
            raise ValueError("No available LLM providers")
        
        # Check cache
        if use_cache:
            cache_key = self._build_cache_key(messages, tools, selected_provider)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                cached_response.cached = True
                return cached_response
        
        # Estimate tokens and check budget
        estimated_tokens = self._estimate_tokens(messages, tools)
        if not self.budget_manager.check_budget(selected_provider, estimated_tokens):
            # Try fallback provider
            fallback = self._get_fallback_provider(selected_provider)
            if fallback and self.budget_manager.check_budget(fallback, estimated_tokens):
                selected_provider = fallback
            else:
                raise ValueError(f"Token budget exceeded for {selected_provider}")
        
        # Generate response
        provider_instance = self.providers[selected_provider]
        
        try:
            response = await provider_instance.generate(messages, tools, **kwargs)
            
            # Record usage
            self.budget_manager.record_usage(selected_provider, response.usage.total_tokens)
            
            # Cache response
            if use_cache:
                self._cache_response(cache_key, response)
            
            # Emit usage event
            await self.event_bus.publish("llm.usage", {
                "provider": selected_provider.value,
                "model": response.model,
                "usage": asdict(response.usage),
                "timestamp": datetime.now().isoformat()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed with {selected_provider}: {e}")
            
            # Try fallback
            fallback = self._get_fallback_provider(selected_provider)
            if fallback and fallback != selected_provider:
                logger.info(f"Attempting fallback to {fallback}")
                return await self.generate(messages, tools, fallback, use_cache, **kwargs)
            
            raise
    
    def _select_best_provider(
        self, 
        messages: List[LLMMessage], 
        tools: Optional[List[Dict]]
    ) -> Optional[LLMProvider]:
        """Select best provider based on capabilities and availability."""
        
        # If tools are needed, prefer providers with function calling
        if tools:
            for provider in [LLMProvider.OPENAI, LLMProvider.AZURE, LLMProvider.OPENROUTER]:
                if provider in self.providers:
                    return provider
        
        # Default priority order
        priority = [
            LLMProvider.OPENAI,
            LLMProvider.AZURE, 
            LLMProvider.OPENROUTER,
            LLMProvider.LOCAL
        ]
        
        for provider in priority:
            if provider in self.providers:
                return provider
        
        return None
    
    def _get_fallback_provider(self, current: LLMProvider) -> Optional[LLMProvider]:
        """Get fallback provider."""
        fallback_map = {
            LLMProvider.OPENAI: LLMProvider.OPENROUTER,
            LLMProvider.AZURE: LLMProvider.OPENAI,
            LLMProvider.OPENROUTER: LLMProvider.LOCAL,
            LLMProvider.LOCAL: None
        }
        
        fallback = fallback_map.get(current)
        return fallback if fallback and fallback in self.providers else None
    
    def _estimate_tokens(
        self, 
        messages: List[LLMMessage], 
        tools: Optional[List[Dict]]
    ) -> int:
        """Estimate total tokens for request."""
        total = 0
        
        for message in messages:
            total += len(message.content.split()) * 1.3  # Rough estimate
        
        if tools:
            tools_str = json.dumps(tools)
            total += len(tools_str.split()) * 1.3
        
        # Add estimated completion tokens
        total += 500
        
        return int(total)
    
    def _build_cache_key(
        self,
        messages: List[LLMMessage], 
        tools: Optional[List[Dict]],
        provider: LLMProvider
    ) -> str:
        """Build cache key for request."""
        content = ""
        for msg in messages:
            content += f"{msg.role.value}:{msg.content}"
        
        if tools:
            content += json.dumps(tools, sort_keys=True)
        
        content += provider.value
        
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if still valid."""
        if cache_key not in self.response_cache:
            return None
        
        response, timestamp = self.response_cache[cache_key]
        if datetime.now() - timestamp > self.cache_ttl:
            del self.response_cache[cache_key]
            return None
        
        return response
    
    def _cache_response(self, cache_key: str, response: LLMResponse):
        """Cache response with timestamp."""
        self.response_cache[cache_key] = (response, datetime.now())
        
        # Clean old entries
        if len(self.response_cache) > 100:
            oldest_key = min(
                self.response_cache.keys(),
                key=lambda k: self.response_cache[k][1]
            )
            del self.response_cache[oldest_key]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "hourly_usage": dict(self.budget_manager.hourly_usage),
            "daily_usage": dict(self.budget_manager.daily_usage),
            "available_providers": list(self.providers.keys()),
            "cache_size": len(self.response_cache)
        }


# Global orchestrator instance
_orchestrator: Optional[LLMOrchestrator] = None


def get_orchestrator() -> LLMOrchestrator:
    """Get global LLM orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        from .event_bus import get_event_bus
        _orchestrator = LLMOrchestrator(get_event_bus())
    return _orchestrator