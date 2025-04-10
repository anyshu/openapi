# filepath: /Users/hc/working/temp/openapi/kimi_vl_service.py
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from urllib.request import urlopen
import base64
import io
from typing import Iterator, Tuple
from threading import Thread
from transformers import TextIteratorStreamer
import time

model = None
processor = None

def initialize_model(model_path: str):
    global model, processor
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    

def extract_image_and_text(messages: list) -> Tuple[str, str]:
    text_prompt = ""
    image_url = None
    
    for msg in messages:
        if msg.get("role") == "user" and "content" in msg:
            content = msg["content"]
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url")
                    elif item.get("type") == "text":
                        text_prompt += item.get("text", "") + " "
            elif isinstance(content, str):
                text_prompt += content + " "
    
    if not image_url:
        raise ValueError("No image URL found in messages")
        
    return image_url, text_prompt.strip() or "What is in this image?"

def process_vl_request(image_path_or_url: str, text_prompt: str):
    try:
        if image_path_or_url.startswith('data:image'):
            # Handle base64 encoded image
            image_data = image_path_or_url.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        elif image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
            image = Image.open(urlopen(image_path_or_url))
        else:
            image = Image.open(image_path_or_url)

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path_or_url},
                {"type": "text", "text": text_prompt}
            ]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

def process_vl_request_stream(image_path_or_url: str, text_prompt: str) -> Iterator[dict]:
    try:
        if image_path_or_url.startswith('data:image'):
            image_data = image_path_or_url.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        elif image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
            image = Image.open(urlopen(image_path_or_url))
        else:
            image = Image.open(image_path_or_url)

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path_or_url},
                {"type": "text", "text": text_prompt}
            ]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(inputs, max_new_tokens=512, streamer=streamer)
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # Send initial response
        yield {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "kimi-vl",
            "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]
        }
        
        for new_text in streamer:
            if new_text:  # Only yield if there's actual content
                yield {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "kimi-vl",
                    "choices": [{"delta": {"content": new_text}, "finish_reason": None}]
                }
        
        # Send completion message
        yield {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "kimi-vl",
            "choices": [{"delta": {}, "finish_reason": "stop"}]
        }
    except Exception as e:
        yield {"error": str(e)}