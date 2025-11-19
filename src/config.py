"""
Simple configuration system with hardcoded model lists and basic Python types.
"""

# Hardcoded model list
AVAILABLE_MODELS = [
    "meta-llama/llama-3.2-3b-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "meta-llama/llama-3.1-8b-instruct",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "meta-llama/llama-3.2-11b-vision-instruct",
    "google/gemma-3-12b-it",
    "qwen/qwen-2.5-14b-instruct",
    "google/gemma-3-27b-it",
    "qwen/qwen3-coder-30b-a3b-instruct",
]

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    "temperature": 0.1,
    "max_tokens": 1024,
    "retry_count": 3
}

# Strategy configurations with ablation support
STRATEGY_CONFIGS = {
    "multi_stage": {
        "use_analysis": True,
        "use_verification": True,
        "analysis_temp": 0.2,
        "generation_temp": 0.1,
        "verification_temp": 0.05
    },
    "multi_stage_no_analysis": {
        "use_analysis": False,
        "use_verification": True,
        "generation_temp": 0.1,
        "verification_temp": 0.05
    },
    "multi_stage_no_verification": {
        "use_analysis": True,
        "use_verification": False,
        "analysis_temp": 0.2,
        "generation_temp": 0.1
    },
    "multi_stage_simple": {
        "use_analysis": False,
        "use_verification": False,
        "generation_temp": 0.1
    }
}

def get_quick_test_config() -> dict:
    """Get config for quick testing."""
    return {
        "name": "quick_test",
        "models": [AVAILABLE_MODELS[0]],  # Just first model
        "strategy": "multi_stage",
        "max_examples": 10,
        "spider_path": "spider",
        "split": "dev"
    }

def get_model_comparison_config() -> dict:
    """Get config for comparing all models."""
    return {
        "name": "model_comparison",
        "models": AVAILABLE_MODELS,  # All models
        "strategy": "multi_stage",
        "max_examples": 50,
        "spider_path": "spider",
        "split": "dev"
    }

def get_strategy_ablation_config() -> dict:
    """Get config for strategy ablation study."""
    return {
        "name": "strategy_ablation",
        "models": [AVAILABLE_MODELS[0]],  # Single model
        "strategies": list(STRATEGY_CONFIGS.keys()),  # All strategy variants
        "max_examples": 100,
        "spider_path": "spider",
        "split": "dev"
    }

def get_comprehensive_config() -> dict:
    """Get config for comprehensive evaluation."""
    return {
        "name": "comprehensive",
        "models": AVAILABLE_MODELS,
        "strategies": list(STRATEGY_CONFIGS.keys()),
        "max_examples": None,  # All examples
        "spider_path": "spider",
        "split": "dev"
    }

def get_model_params(model_name: str, **overrides) -> dict:
    """Get model parameters with optional overrides."""
    params = DEFAULT_MODEL_PARAMS.copy()
    params["model"] = model_name
    params.update(overrides)
    return params

def get_strategy_config(strategy_name: str) -> dict:
    """Get strategy configuration."""
    if strategy_name not in STRATEGY_CONFIGS:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_CONFIGS.keys())}")
    return STRATEGY_CONFIGS[strategy_name].copy()

def list_available_models() -> list:
    """List all available models."""
    return AVAILABLE_MODELS.copy()

def list_available_strategies() -> list:
    """List all available strategy configurations."""
    return list(STRATEGY_CONFIGS.keys())

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
        if config["strategy"] not in STRATEGY_CONFIGS:
            print(f"Error: Unknown strategy: {config['strategy']}")
            return False
    elif "strategies" in config:
        for strategy in config["strategies"]:
            if strategy not in STRATEGY_CONFIGS:
                print(f"Error: Unknown strategy: {strategy}")
                return False

    return True