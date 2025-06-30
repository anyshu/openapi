from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import torch
import time
from PIL import Image
import io
import base64

# 启用 TF32 以获得更好的性能（适用于 Ampere 架构 GPU）
torch.set_float32_matmul_precision('high')

model = None
processor = None

def initialize_model(model_path: str):
    """初始化 Google Gemma 3n 多模态模型"""
    global model, processor
    try:
        # 加载 processor
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 加载 Gemma 3n 条件生成模型
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        
        print(f"成功加载 Gemma 3n 模型: {model_path}")
        print(f"模型架构: {model.config.model_type}")
        print(f"支持多模态输入: 文本 + 图像")
        
    except Exception as e:
        raise RuntimeError(f"加载 Gemma 3n 模型失败: {str(e)}")

def convert_openai_to_gemma3n_format(messages):
    """将 OpenAI 格式的消息转换为 Gemma 3n 格式"""
    converted_messages = []
    
    for message in messages:
        converted_message = {
            "role": message["role"],
            "content": []
        }
        
        content = message.get("content", [])
        if isinstance(content, str):
            # 纯文本消息
            converted_message["content"].append({
                "type": "text", 
                "text": content
            })
        elif isinstance(content, list):
            # 多模态消息
            for item in content:
                if item.get("type") == "text":
                    converted_message["content"].append({
                        "type": "text",
                        "text": item["text"]
                    })
                elif item.get("type") == "image_url":
                    image_url = item["image_url"]["url"]
                    
                    # 处理 base64 编码的图像
                    if image_url.startswith("data:image"):
                        base64_data = image_url.split(",")[1]
                        image_data = base64.b64decode(base64_data)
                        image = Image.open(io.BytesIO(image_data))
                        
                        converted_message["content"].append({
                            "type": "image",
                            "image": image
                        })
                    else:
                        raise ValueError("暂不支持远程图像 URL，请使用 base64 编码的图像")
        
        converted_messages.append(converted_message)
    
    return converted_messages

def generate_chat_response(request: dict) -> dict:
    """生成聊天响应（支持多模态输入）"""
    try:
        messages = request["messages"]
        max_tokens = request.get("max_tokens", 1024)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 0.9)
        do_sample = temperature > 0
        
        # 转换消息格式
        converted_messages = convert_openai_to_gemma3n_format(messages)
        
        # 使用 processor 应用聊天模板
        inputs = processor.apply_chat_template(
            converted_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # 生成响应
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
            )
            generation = generation[0][input_len:]
        
        # 解码响应
        response_text = processor.decode(generation, skip_special_tokens=True)
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request["model"],
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text.strip()
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": len(generation),
                "total_tokens": input_len + len(generation)
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
        do_sample = temperature > 0
        
        # 转换消息格式
        converted_messages = convert_openai_to_gemma3n_format(messages)
        
        # 使用 processor 应用聊天模板
        inputs = processor.apply_chat_template(
            converted_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # 生成响应
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
            )
            generation = generation[0][input_len:]
        
        # 解码响应
        response_text = processor.decode(generation, skip_special_tokens=True)
        
        # 模拟流式返回
        words = response_text.split()
        for i, word in enumerate(words):
            chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request["model"],
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
            "model": request["model"],
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