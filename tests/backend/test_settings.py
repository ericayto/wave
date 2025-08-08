"""
Test the configuration system and TOML settings loading.
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wave_backend.config.settings import Settings, get_settings


def test_default_settings():
    """Test default settings values."""
    settings = Settings()
    
    # Core settings
    assert settings.core.base_currency == "USD"
    assert settings.core.mode == "paper"
    
    # Risk settings
    assert settings.risk.max_position_pct == 0.25
    assert settings.risk.daily_loss_limit_pct == 2.0
    assert settings.risk.max_orders_per_hour == 6
    
    # UI settings
    assert settings.ui.port == 5173
    assert settings.ui.show_thinking_feed is True


def test_toml_config_loading():
    """Test loading configuration from TOML file."""
    # Create a temporary TOML file
    toml_content = """
[core]
base_currency = "EUR"
mode = "paper"

[risk]
max_position_pct = 0.15
daily_loss_limit_pct = 1.5
max_orders_per_hour = 10

[llm]
provider = "openai"
model = "gpt-4"
planning_enabled = true
hourly_token_budget = 25000

[exchanges.kraken]
api_key = "test_key"
api_secret = "test_secret"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        temp_file = f.name
    
    try:
        # Set the config file path and load settings
        os.environ['WAVE_CONFIG'] = temp_file
        settings = get_settings()
        
        # Check that values were loaded from TOML
        assert settings.core.base_currency == "EUR"
        assert settings.risk.max_position_pct == 0.15
        assert settings.risk.daily_loss_limit_pct == 1.5
        assert settings.risk.max_orders_per_hour == 10
        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4"
        assert settings.llm.planning_enabled is True
        assert settings.llm.hourly_token_budget == 25000
        
    finally:
        # Clean up
        os.unlink(temp_file)
        if 'WAVE_CONFIG' in os.environ:
            del os.environ['WAVE_CONFIG']


def test_environment_variable_resolution():
    """Test that environment variables are resolved in config."""
    # Set environment variables
    os.environ['TEST_KRAKEN_KEY'] = 'env_resolved_key'
    os.environ['TEST_KRAKEN_SECRET'] = 'env_resolved_secret'
    
    toml_content = """
[exchanges.kraken]
api_key = "env:TEST_KRAKEN_KEY"
api_secret = "env:TEST_KRAKEN_SECRET"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        temp_file = f.name
    
    try:
        os.environ['WAVE_CONFIG'] = temp_file
        settings = get_settings()
        
        # Check environment variables were resolved
        assert settings.exchanges.kraken.api_key == "env_resolved_key"
        assert settings.exchanges.kraken.api_secret == "env_resolved_secret"
        
    finally:
        # Clean up
        os.unlink(temp_file)
        for key in ['WAVE_CONFIG', 'TEST_KRAKEN_KEY', 'TEST_KRAKEN_SECRET']:
            if key in os.environ:
                del os.environ[key]


def test_memory_settings():
    """Test memory and context management settings."""
    toml_content = """
[memory]
max_context_tokens = 100000
target_window_tokens = 20000
rag_top_k = 8
summarize_every_events = 30
summary_target_tokens = 1500
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        temp_file = f.name
    
    try:
        os.environ['WAVE_CONFIG'] = temp_file
        settings = get_settings()
        
        assert settings.memory.max_context_tokens == 100000
        assert settings.memory.target_window_tokens == 20000
        assert settings.memory.rag_top_k == 8
        assert settings.memory.summarize_every_events == 30
        assert settings.memory.summary_target_tokens == 1500
        
    finally:
        os.unlink(temp_file)
        if 'WAVE_CONFIG' in os.environ:
            del os.environ['WAVE_CONFIG']


def test_llm_provider_settings():
    """Test LLM provider specific settings."""
    toml_content = """
[llm]
provider = "openrouter"
model = "gpt-4o-mini"
max_tokens = 1024
temperature = 0.2
use_local_summarizer = true

[llm.providers.openai]
api_key = "env:OPENAI_API_KEY"
base_url = "https://api.openai.com/v1"

[llm.providers.openrouter]
api_key = "env:OPENROUTER_API_KEY"
base_url = "https://openrouter.ai/api/v1"

[llm.providers.local]
base_url = "http://localhost:11434"
model = "llama3.1:8b"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        temp_file = f.name
    
    try:
        os.environ['WAVE_CONFIG'] = temp_file
        settings = get_settings()
        
        # Check LLM settings
        assert settings.llm.provider == "openrouter"
        assert settings.llm.model == "gpt-4o-mini"
        assert settings.llm.max_tokens == 1024
        assert settings.llm.temperature == 0.2
        assert settings.llm.use_local_summarizer is True
        
        # Check provider configurations
        assert settings.llm.providers.openai.base_url == "https://api.openai.com/v1"
        assert settings.llm.providers.openrouter.base_url == "https://openrouter.ai/api/v1"
        assert settings.llm.providers.local.base_url == "http://localhost:11434"
        assert settings.llm.providers.local.model == "llama3.1:8b"
        
    finally:
        os.unlink(temp_file)
        if 'WAVE_CONFIG' in os.environ:
            del os.environ['WAVE_CONFIG']


def test_settings_singleton():
    """Test that get_settings returns the same instance."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2


def test_invalid_toml_handling():
    """Test handling of invalid TOML files."""
    invalid_toml = """
[core]
invalid_toml_syntax = 
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(invalid_toml)
        temp_file = f.name
    
    try:
        os.environ['WAVE_CONFIG'] = temp_file
        # This should either raise an exception or fall back to defaults
        # depending on implementation
        settings = get_settings()
        # If it doesn't raise, at least verify core settings exist
        assert hasattr(settings, 'core')
        
    finally:
        os.unlink(temp_file)
        if 'WAVE_CONFIG' in os.environ:
            del os.environ['WAVE_CONFIG']