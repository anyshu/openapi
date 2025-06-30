from transformers import AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration
import torch
import time
from PIL import Image
import io
import base64

model = None
tokenizer = None
processor = None

def initialize_model(model_path: str):
    """初始化 Google Gemma 3n 多模态模型"""
    global model, tokenizer, processor
    try:
        # 加载 tokenizer 和 processor
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 加载 Gemma 3n 条件生成模型
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # Gemma 3n 特定配置
            low_cpu_mem_usage=True,
        )
        
        print(f"成功加载 Gemma 3n 模型: {model_path}")
        print(f"模型架构: {model.config.model_type}")
        print(f"支持多模态输入: 文本 + 图像")
        
    except Exception as e:
        raise RuntimeError(f"加载 Gemma 3n 模型失败: {str(e)}")

def process_image_content(content_item):
    """处理图像内容"""
    if content_item.get("type") == "image_url":
        image_url = content_item["image_url"]["url"]
        
        # 如果是 base64 编码的图像
        if image_url.startswith("data:image"):
            # 提取 base64 数据
            base64_data = image_url.split(",")[1]
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
        else:
            # 如果是 URL（这里简化处理，实际可能需要下载）
            raise ValueError("暂不支持远程图像 URL，请使用 base64 编码的图像")
        
        return image
    return None

def generate_chat_response(request: dict) -> dict:
    """生成聊天响应（支持多模态输入）"""
    try:
        messages = request["messages"]
        max_tokens = request.get("max_tokens", 1024)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 0.9)
        
        # 处理多模态消息
        text_content = []
        images = []
        
        for message in messages:
            if message["role"] in ["user", "assistant"]:
                content = message.get("content", [])
                if isinstance(content, str):
                    text_content.append(f"{message['role']}: {content}")
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            text_content.append(f"{message['role']}: {item['text']}")
                        elif item.get("type") == "image_url":
                            image = process_image_content(item)
                            if image is not None:
                                images.append(image)
        
        # 构建输入文本
        conversation_text = "\n".join(text_content)
        
        # 使用 processor 处理多模态输入
        if images:
            # 多模态输入（文本 + 图像）
            inputs = processor(
                text=conversation_text,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(model.device)
        else:
            # 纯文本输入
            inputs = processor(
                text=conversation_text,
                return_tensors="pt",
                padding=True
            ).to(model.device)
        
        # 生成配置
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # 生成响应
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config
            )
        
        # 解码响应
        input_length = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated_tokens = outputs[0][input_length:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "google/gemma-3n-e4b-it",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text.strip()
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": len(generated_tokens),
                "total_tokens": input_length + len(generated_tokens)
            }
        }
        
    except Exception as e:
        raise RuntimeError(f"Gemma 3n 生成响应失败: {str(e)}")

def generate_stream_response(request: dict):
    """生成流式响应（支持多模态输入）"""
    try:
        messages = request["messages"]
        max_tokens = request.get("max_tokens", 1024)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 0.9)
        
        # 处理多模态消息（与上面相同的逻辑）
        text_content = []
        images = []
        
        for message in messages:
            if message["role"] in ["user", "assistant"]:
                content = message.get("content", [])
                if isinstance(content, str):
                    text_content.append(f"{message['role']}: {content}")
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            text_content.append(f"{message['role']}: {item['text']}")
                        elif item.get("type") == "image_url":
                            image = process_image_content(item)
                            if image is not None:
                                images.append(image)
        
        conversation_text = "\n".join(text_content)
        
        # 使用 processor 处理输入
        if images:
            inputs = processor(
                text=conversation_text,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(model.device)
        else:
            inputs = processor(
                text=conversation_text,
                return_tensors="pt",
                padding=True
            ).to(model.device)
        
        # 流式生成配置
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # 生成并流式返回
        input_length = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config
            )
        
        generated_tokens = outputs[0][input_length:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 模拟流式返回
        words = response_text.split()
        for i, word in enumerate(words):
            chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "google/gemma-3n-e4b-it",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": word + " " if i < len(words) - 1 else word
                    },
                    "finish_reason": None
                }]
            }
            yield f"data: {str(chunk)}\n\n"
        
        # 结束标记
        end_chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "google/gemma-3n-e4b-it",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {str(end_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_msg = f"Gemma 3n 流式生成失败: {str(e)}"
        yield f"data: {{'error': '{error_msg}'}}\n\n" 