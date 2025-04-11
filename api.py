from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional, AsyncGenerator
from pydantic import BaseModel
import time
import json
import asyncio
import argparse
import importlib
import os
from model_config import MODEL_CONFIGS, get_handler_for_model

app = FastAPI()
loaded_models = {}

def load_model_handler(model_path: str, model_name: str = None):
    """根据模型路径和名称加载对应的处理器"""
    try:
        if model_name is None:
            model_name = os.path.basename(model_path)
        
        handler_name = get_handler_for_model(model_name)
        handler_module = importlib.import_module(handler_name)
        handler_module.initialize_model(model_path)
        loaded_models[model_name] = handler_module
        return model_name
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "api_error",
                "param": None,
                "code": f"error_{exc.status_code}"
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "server_error",
                "param": None,
                "code": "internal_server_error"
            }
        }
    )

def format_vl_response(result: dict, model_name: str) -> dict:
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result["response"]
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

async def async_generator(sync_generator):
    for item in sync_generator:
        if isinstance(item, dict):
            yield f"data: {json.dumps(item)}\n\n"
        await asyncio.sleep(0)
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completion(request: dict = Body(...)):
    try:
        if not request or "messages" not in request:
            raise HTTPException(status_code=400, detail="messages field is required")

        model = request.get("model", "")
        if model not in loaded_models:
            raise HTTPException(status_code=400, detail=f"Model {model} not loaded")

        handler = loaded_models[model]
        stream_mode = request.get("stream", False)

        if any(msg.get("content") and isinstance(msg["content"], list) for msg in request["messages"]):
            # Handle vision request
            image_url, text_prompt = handler.extract_image_and_text(request["messages"])
            if stream_mode:
                generator = handler.process_vl_request_stream(image_url, text_prompt)
                return StreamingResponse(
                    async_generator(generator), 
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Transfer-Encoding": "chunked"
                    }
                )
            result = handler.process_vl_request(image_url, text_prompt)
            return format_vl_response(result, model)
        else:
            # Handle text-only chat request
            return handler.generate_chat_response(request)

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to the model")
    parser.add_argument("--model-name", help="Model name (optional, defaults to path basename)")
    args = parser.parse_args()
    
    model_name = load_model_handler(args.model_path, args.model_name)
    print(f"Loaded model {model_name} from {args.model_path}")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12200)
