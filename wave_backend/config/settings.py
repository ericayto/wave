"""
Wave Configuration Management
Loads and validates TOML configuration files.
"""

import os
import toml
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from functools import lru_cache

class CoreConfig(BaseModel):
    base_currency: str = "USD"
    mode: str = "paper"
    data_dir: str = "./data"
    log_level: str = "INFO"

class ExchangeConfig(BaseModel):
    api_key: str
    api_secret: str
    sandbox: bool = True

class LLMProviderConfig(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    endpoint: Optional[str] = None
    model: str = "gpt-4o-mini"
    hourly_token_budget: int = 50000
    daily_token_budget: int = 500000
    api_version: Optional[str] = None

class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    temperature: float = 0.1
    hourly_token_budget: int = 50000
    daily_token_budget: int = 500000
    planning_enabled: bool = True
    planning_interval_seconds: int = 300
    use_cache: bool = True
    cache_ttl_minutes: int = 15
    providers: Dict[str, LLMProviderConfig] = {}

class MemoryConfig(BaseModel):
    max_context_tokens: int = 128000
    target_window_tokens: int = 24000
    rag_top_k: int = 6
    summarize_every_events: int = 25
    summarize_every_minutes: int = 15
    summary_target_tokens: int = 1200
    use_local_summarizer: bool = True

class RiskConfig(BaseModel):
    max_position_pct: float = 0.25
    daily_loss_limit_pct: float = 2.0
    max_orders_per_hour: int = 6
    circuit_breaker_spread_bps: int = 50

class UIConfig(BaseModel):
    port: int = 5173
    api_port: int = 8080
    show_thinking_feed: bool = True
    theme: str = "ocean"

class DatabaseConfig(BaseModel):
    url: str = "sqlite:///./data/wave.db"
    echo: bool = False

class StrategiesConfig(BaseModel):
    default_symbols: list[str] = ["BTC/USDT", "ETH/USDT"]
    default_timeframe: str = "5m"
    max_active_strategies: int = 5

class Settings(BaseModel):
    core: CoreConfig = CoreConfig()
    exchanges: Dict[str, ExchangeConfig] = {}
    llm: LLMConfig = LLMConfig()
    memory: MemoryConfig = MemoryConfig()
    risk: RiskConfig = RiskConfig()
    ui: UIConfig = UIConfig()
    database: DatabaseConfig = DatabaseConfig()
    strategies: StrategiesConfig = StrategiesConfig()

def resolve_env_var(value: str) -> str:
    """Resolve environment variables in config values."""
    if isinstance(value, str) and value.startswith("env:"):
        env_var = value[4:]  # Remove "env:" prefix
        return os.getenv(env_var, "")
    return value

def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load and parse TOML configuration file."""
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config_data = toml.load(f)
        
        # Recursively resolve environment variables
        def resolve_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    result[key] = resolve_dict(value)
                elif isinstance(value, str):
                    result[key] = resolve_env_var(value)
                else:
                    result[key] = value
            return result
        
        return resolve_dict(config_data)
    except Exception as e:
        print(f"Warning: Failed to load config file {config_path}: {e}")
        return {}

@lru_cache()
def get_settings() -> Settings:
    """Get application settings, loading from TOML config file."""
    # Look for config file in multiple locations
    config_paths = [
        Path("./config/wave.toml"),
        Path("./wave.toml"),
        Path.home() / ".wave" / "config.toml",
    ]
    
    config_data = {}
    for config_path in config_paths:
        if config_path.exists():
            config_data = load_config_file(config_path)
            print(f"📝 Loaded config from: {config_path}")
            break
    
    if not config_data:
        print("📝 Using default configuration")
    
    return Settings(**config_data)

# Global settings instance for easy imports
settings = get_settings()