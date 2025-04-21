MODEL_CONFIGS = {
    "kimi-vl": "kimi_vl_service",
    "dream": "dream_service",
    "GLM-Z1-32B-0414": "glm4_handler"
}

def get_handler_for_model(model_path: str) -> str:
    """根据模型路径返回对应的处理器名称"""
    model_path = model_path.lower()
    for model_name, handler in MODEL_CONFIGS.items():
        if model_name in model_path:
            return handler
    raise ValueError(f"No handler found for model path: {model_path}")

def get_handler_by_name(model_name: str) -> str:
    """根据模型名称返回对应的处理器名称"""
    model_name = model_name.lower()
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    raise ValueError(f"No handler found for model: {model_name}")