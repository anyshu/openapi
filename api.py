from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional, AsyncGenerator
from pydantic import BaseModel
from dream_service import generate_chat_response, ChatMessage, ChatCompletionRequest
from kimi_vl_service import process_vl_request, process_vl_request_stream
import time
import json
import asyncio

app = FastAPI()

class VLCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    image: UploadFile
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95

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

def format_vl_response(result: dict) -> dict:
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "kimi-vl",
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
    """Convert sync generator to async and format as SSE"""
    for item in sync_generator:
        if isinstance(item, dict):
            yield f"data: {json.dumps(item)}\n\n"
        await asyncio.sleep(0)
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completion(request: dict = Body(...)):
    try:
        if not request or "messages" not in request:
            raise HTTPException(status_code=400, detail="messages field is required in the request body")

        messages = request["messages"]
        image_url = None
        text_prompt = ""
        stream_mode = request.get("stream", False)

        # Extract image URL and text from messages
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

        if image_url:
            if stream_mode:
                generator = process_vl_request_stream(image_url, text_prompt.strip() or "What is in this image?")
                return StreamingResponse(
                    async_generator(generator), 
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Transfer-Encoding": "chunked"
                    }
                )
            result = process_vl_request(image_url, text_prompt.strip() or "What is in this image?")
            return format_vl_response(result)

        # Handle text-only chat request
        chat_request = ChatCompletionRequest(
            messages=[ChatMessage(**msg) for msg in messages],
            max_tokens=request.get("max_tokens", 512),
            temperature=request.get("temperature", 0.7),
            top_p=request.get("top_p", 0.95)
        )
        return generate_chat_response(chat_request)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12200)
