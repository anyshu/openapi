MODEL_TYPE_MAPPING = {
    "kimi-vl": "kimi_vl_service",  # 匹配所有包含 kimi-vl 的模型
    "dream": "dream_service",       # 匹配所有包含 dream 的模型
}

def get_handler_for_model(model_path: str) -> str:
    """根据模型路径返回对应的处理器名称"""
    model_path = model_path.lower()
    for key_word, handler in MODEL_TYPE_MAPPING.items():
        if key_word in model_path:
            return handler
    raise ValueError(f"No handler found for model path: {model_path}")