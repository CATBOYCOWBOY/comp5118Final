"""
Spider2-lite configuration system with hardcoded model lists and basic Python types.
"""

# Hardcoded model list
AVAILABLE_MODELS = [
    "meta-llama/llama-3.2-3b-instruct",
    "qwen/qwen3-4b-fp8",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen3-8b-fp8",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "qwen/qwen3-coder-30b-a3b-instruct",
    "qwen/qwen3-32b-fp8",
    "deepseek/deepseek-r1-distill-qwen-14b",
    "qwen/qwen2.5-7b-instruct",
    "openai/gpt-oss-20b"
]

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    "temperature": 0.1,
    "max_tokens": 8192,
    "retry_count": 3
}

# Spider2 Strategy configuration
STRATEGY_CONFIG = {
    "spider2_basic": {
        "generation_temp": 0.1
    }
}

def get_spider2_quick_test_config() -> dict:
    """Get config for Spider2-lite quick testing."""
    return {
        "name": "spider2_quick_test",
        "models": [AVAILABLE_MODELS[0]],
        "strategy": "spider2_basic",
        "max_examples": 10,
        "spider2_path": "Spider2",
        "dataset_type": "spider2"
    }

def get_spider2_comparison_config() -> dict:
    """Get config for Spider2-lite model comparison."""
    return {
        "name": "spider2_model_comparison",
        "models": AVAILABLE_MODELS,
        "strategy": "spider2_basic",
        "max_examples": 50,
        "spider2_path": "Spider2",
        "dataset_type": "spider2"
    }

def get_spider2_comprehensive_config() -> dict:
    """Get config for comprehensive Spider2-lite evaluation."""
    return {
        "name": "spider2_comprehensive",
        "models": AVAILABLE_MODELS,
        "strategy": "spider2_basic",
        "max_examples": None,  # All examples
        "spider2_path": "Spider2",
        "dataset_type": "spider2"
    }

def get_model_params(model_name: str, **overrides) -> dict:
    """Get model parameters with optional overrides."""
    params = DEFAULT_MODEL_PARAMS.copy()
    params["model"] = model_name
    params.update(overrides)
    return params

def get_strategy_config(strategy_name: str = "spider2_basic") -> dict:
    """Get strategy configuration."""
    if strategy_name not in STRATEGY_CONFIG:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_CONFIG.keys())}")
    return STRATEGY_CONFIG[strategy_name].copy()

def list_available_models() -> list:
    """List all available models."""
    return AVAILABLE_MODELS.copy()

def validate_config(config: dict) -> bool:
    """Basic config validation."""
    if "models" not in config or not config["models"]:
        print("Error: No models specified")
        return False

    for model in config["models"]:
        if model not in AVAILABLE_MODELS:
            print(f"Error: Unknown model: {model}")
            return False

    if "strategy" in config:
        if config["strategy"] not in STRATEGY_CONFIG:
            print(f"Error: Unknown strategy: {config['strategy']}")
            return False

    return True
