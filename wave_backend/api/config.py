"""
Configuration API endpoints for Wave
Handles onboarding wizard configuration saving and loading
"""

import os
import toml
from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/config", tags=["config"])

class OnboardingConfig(BaseModel):
    """Configuration data from onboarding wizard"""
    exchange: Dict[str, Any]
    llmProvider: Dict[str, Any] 
    risk: Dict[str, Any]
    trading: Dict[str, Any]

def get_config_path() -> Path:
    """Get path to configuration file"""
    return Path("./config/wave.toml")

def load_existing_config() -> Dict[str, Any]:
    """Load existing TOML configuration"""
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path, 'r') as f:
            return toml.load(f)
    return {}

def save_config_to_toml(config_data: Dict[str, Any]) -> None:
    """Save configuration data to TOML file"""
    config_path = get_config_path()
    
    # Ensure config directory exists
    config_path.parent.mkdir(exist_ok=True)
    
    # Load existing config to preserve other settings
    existing_config = load_existing_config()
    
    # Update with new configuration
    # Exchange settings
    if 'exchange' in config_data:
        existing_config.setdefault('exchanges', {})
        existing_config['exchanges']['kraken'] = {
            'api_key': f"env:KRAKEN_KEY",  # Store as env var reference
            'api_secret': f"env:KRAKEN_SECRET",  # Store as env var reference  
            'sandbox': config_data['exchange'].get('sandboxMode', True)
        }
        
        # Set environment variables (in production, these would be handled securely)
        os.environ['KRAKEN_KEY'] = config_data['exchange'].get('krakenApiKey', '')
        os.environ['KRAKEN_SECRET'] = config_data['exchange'].get('krakenApiSecret', '')
    
    # LLM Provider settings
    if 'llmProvider' in config_data:
        llm_data = config_data['llmProvider']
        existing_config['llm'] = {
            'provider': llm_data.get('provider', 'openai'),
            'model': llm_data.get('model', 'gpt-4o-mini'),
            'max_tokens': 1000,
            'temperature': 0.1,
            'hourly_token_budget': llm_data.get('hourlyTokenBudget', 50000),
            'daily_token_budget': llm_data.get('dailyTokenBudget', 500000),
            'planning_enabled': True,
            'planning_interval_seconds': 300,
            'use_cache': True,
            'cache_ttl_minutes': 15,
            'providers': {}
        }
        
        # Provider-specific settings
        provider = llm_data.get('provider', 'openai')
        provider_config = {
            'model': llm_data.get('model', 'gpt-4o-mini'),
            'hourly_token_budget': llm_data.get('hourlyTokenBudget', 50000),
            'daily_token_budget': llm_data.get('dailyTokenBudget', 500000)
        }
        
        if provider == 'openai':
            provider_config.update({
                'api_key': 'env:OPENAI_API_KEY',
                'base_url': 'https://api.openai.com/v1'
            })
            os.environ['OPENAI_API_KEY'] = llm_data.get('apiKey', '')
        elif provider == 'azure':
            provider_config.update({
                'api_key': 'env:AZURE_OPENAI_KEY',
                'endpoint': 'env:AZURE_OPENAI_ENDPOINT',
                'api_version': '2024-02-15-preview'
            })
            os.environ['AZURE_OPENAI_KEY'] = llm_data.get('apiKey', '')
            os.environ['AZURE_OPENAI_ENDPOINT'] = llm_data.get('endpoint', '')
        elif provider == 'openrouter':
            provider_config.update({
                'api_key': 'env:OPENROUTER_API_KEY',
                'base_url': 'https://openrouter.ai/api/v1'
            })
            os.environ['OPENROUTER_API_KEY'] = llm_data.get('apiKey', '')
        elif provider == 'local':
            provider_config.update({
                'base_url': llm_data.get('baseUrl', 'http://localhost:11434')
            })
        
        existing_config['llm']['providers'][provider] = provider_config
    
    # Risk settings
    if 'risk' in config_data:
        risk_data = config_data['risk']
        existing_config['risk'] = {
            'max_position_pct': risk_data.get('maxPositionPct', 0.25),
            'daily_loss_limit_pct': risk_data.get('dailyLossLimitPct', 2.0),
            'max_orders_per_hour': risk_data.get('maxOrdersPerHour', 6),
            'circuit_breaker_spread_bps': risk_data.get('circuitBreakerSpreadBps', 50)
        }
    
    # Trading settings
    if 'trading' in config_data:
        trading_data = config_data['trading']
        existing_config['strategies'] = {
            'default_symbols': trading_data.get('defaultSymbols', ['BTC/USDT', 'ETH/USDT']),
            'default_timeframe': trading_data.get('defaultTimeframe', '5m'),
            'max_active_strategies': trading_data.get('maxActiveStrategies', 5)
        }
        existing_config['core'] = existing_config.get('core', {})
        existing_config['core']['base_currency'] = trading_data.get('baseCurrency', 'USD')
    
    # Save to file
    with open(config_path, 'w') as f:
        toml.dump(existing_config, f)

@router.post("/save-onboarding")
async def save_onboarding_config(config: OnboardingConfig):
    """Save configuration from onboarding wizard"""
    try:
        config_dict = config.dict()
        save_config_to_toml(config_dict)
        return {"success": True, "message": "Configuration saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")

@router.get("/status")
async def get_config_status():
    """Check if user has completed onboarding"""
    config_path = get_config_path()
    
    if not config_path.exists():
        return {"is_configured": False}
    
    try:
        config = load_existing_config()
        
        # Check for required configuration sections
        has_exchange = 'exchanges' in config and 'kraken' in config['exchanges']
        has_llm = 'llm' in config and 'provider' in config['llm']
        has_risk = 'risk' in config
        has_strategies = 'strategies' in config
        
        is_configured = has_exchange and has_llm and has_risk and has_strategies
        
        return {
            "is_configured": is_configured,
            "has_exchange": has_exchange,
            "has_llm": has_llm, 
            "has_risk": has_risk,
            "has_strategies": has_strategies,
            "config_summary": {
                "provider": config.get('llm', {}).get('provider', 'unknown'),
                "exchange_mode": 'paper' if config.get('exchanges', {}).get('kraken', {}).get('sandbox', True) else 'live',
                "risk_profile": 'custom',  # We don't store profile name, just settings
                "trading_pairs": len(config.get('strategies', {}).get('default_symbols', []))
            }
        }
    except Exception as e:
        return {"is_configured": False, "error": str(e)}

@router.get("/current")
async def get_current_config():
    """Get current configuration for display"""
    try:
        config = load_existing_config()
        return {"config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load configuration: {str(e)}")