from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from urllib.request import urlopen
import base64
import io
from typing import Iterator, Tuple
from threading import Thread, Lock, Event
from transformers import TextIteratorStreamer
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from collections import deque

model = None
processor = None

# Global KV cache for attention layers
kv_cache = {
    "short_term": {},  # Short-term cache for current inference
    "long_term": deque(maxlen=10)  # Long-term cache with a fixed size
}

# Lock for managing KV cache updates
kv_cache_lock = Lock()

# Task queue for dynamic batching
task_queue = Queue()

# Thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=4)

# Event to signal batch processing
batch_event = Event()

# Batch processing parameters
BATCH_SIZE = 16
MAX_SEQ_LEN = 512

# Dynamic batching queue
batch_queue = []

# Preallocate memory for batch processing
def preallocate_memory(batch_size: int, max_seq_len: int):
    global preallocated_inputs
    preallocated_inputs = {
        "input_ids": torch.zeros((batch_size, max_seq_len), dtype=torch.long, device="cuda"),
        "attention_mask": torch.zeros((batch_size, max_seq_len), dtype=torch.long, device="cuda"),
    }

def initialize_model(model_path: str):
    global model, processor
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",  # Automatically map layers to available GPUs
        max_memory={i: "70GB" for i in range(torch.cuda.device_count())},  # Limit GPU memory usage
        trust_remote_code=True,
    )
    # Compile the model for optimized inference (requires PyTorch 2.0+)
    model = torch.compile(model)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # Preallocate memory for batch processing
    preallocate_memory(batch_size=16, max_seq_len=512)

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

def manage_kv_cache(inputs):
    """
    Manage KV cache by combining short-term and long-term caches.
    """
    with kv_cache_lock:
        # Merge short-term cache into inputs
        if "short_term" in kv_cache and kv_cache["short_term"]:
            inputs["past_key_values"] = kv_cache["short_term"]

        # Optionally load from long-term cache if available
        if kv_cache["long_term"]:
            inputs["past_key_values"] = kv_cache["long_term"][-1]

    return inputs

def update_kv_cache(new_kv):
    """
    Update KV cache with new values.
    """
    with kv_cache_lock:
        # Update short-term cache
        kv_cache["short_term"] = new_kv

        # Push to long-term cache
        kv_cache["long_term"].append(new_kv)

def dynamic_batching_worker():
    """
    Worker function to process requests in batches.
    """
    global batch_queue
    while True:
        batch_event.wait()  # Wait for the signal to process a batch
        batch_event.clear()

        if not batch_queue:
            continue

        # Collect batch inputs
        with kv_cache_lock:
            batch = batch_queue[:BATCH_SIZE]
            batch_queue = batch_queue[BATCH_SIZE:]

        if not batch:
            continue

        # Prepare batched inputs
        input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)
        attention_mask = torch.cat([item["attention_mask"] for item in batch], dim=0)
        past_key_values = [item["past_key_values"] for item in batch if "past_key_values" in item]

        # Perform inference
        with torch.inference_mode(), torch.autocast("cuda"):
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if past_key_values:
                inputs["past_key_values"] = past_key_values[0]  # Use the first item's cache
            outputs = model.generate(**inputs, max_new_tokens=512)

        # Distribute results back to individual requests
        for i, item in enumerate(batch):
            item["result"] = outputs[i]

def enqueue_request(inputs):
    """
    Enqueue a request for dynamic batching.
    """
    global batch_queue
    with kv_cache_lock:
        batch_queue.append(inputs)
        if len(batch_queue) >= BATCH_SIZE:
            batch_event.set()  # Signal the worker to process the batch

def process_vl_request(image_path_or_url: str, text_prompt: str):
    try:
        # Use autocast for mixed precision to save memory
        autocast_context = torch.autocast("cuda") if torch.cuda.is_available() else nullcontext()
        with torch.inference_mode(), autocast_context:
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
            inputs = {k: v.unsqueeze(0) if v.dim() == 1 else v for k, v in inputs.items()}

            # Manage KV cache
            inputs = manage_kv_cache(inputs)

            # Enqueue the request for dynamic batching
            enqueue_request(inputs)

            # Wait for the result
            while "result" not in inputs:
                time.sleep(0.01)  # Polling for the result

            # Extract the result
            generated_ids = inputs["result"]

            # Update KV cache with new values
            update_kv_cache(model.past_key_values)

            # Explicitly clear unused memory
            torch.cuda.empty_cache()

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return {"response": response}
    except Exception as e:
        return {"error": str(e)}

def process_vl_request_stream(image_path_or_url: str, text_prompt: str) -> Iterator[dict]:
    try:
        # Use autocast for mixed precision to save memory
        autocast_context = torch.autocast("cuda") if torch.cuda.is_available() else nullcontext()
        with torch.inference_mode(), autocast_context:
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
            inputs = {k: v.unsqueeze(0) if v.dim() == 1 else v for k, v in inputs.items()}
            
            # Manage KV cache
            inputs = manage_kv_cache(inputs)

            streamer = TextIteratorStreamer(
                processor, skip_prompt=True, skip_special_tokens=True, timeout=0.1
            )  # Reduce timeout for faster streaming
            generation_kwargs = dict(inputs, max_new_tokens=512, streamer=streamer)
            
            # Submit task to thread pool for concurrent execution
            future = executor.submit(model.generate, **generation_kwargs)

            # Update KV cache with new values
            update_kv_cache(model.past_key_values)

            # Explicitly clear unused memory
            torch.cuda.empty_cache()

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