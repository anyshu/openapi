from fastapi import FastAPI, UploadFile, File, Form
from typing import List, Optional
from pydantic import BaseModel
from dream_service import generate_chat_response, ChatMessage, ChatCompletionRequest
from kimi_vl_service import process_vl_request

app = FastAPI()

class VLCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    image: UploadFile
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95

@app.post("/v1/chat/completions")
async def chat_completion(request: dict = None, image_url: Optional[str] = Form(None), image: Optional[UploadFile] = File(None)):
    # Handle vision/language requests with images
    if image_url or image:
        file_location_or_url = None
        if image_url:
            file_location_or_url = image_url
        elif image:
            file_location = f"/tmp/{image.filename}"
            with open(file_location, "wb") as f:
                f.write(image.file.read())
            file_location_or_url = file_location
            
        # Extract text from messages
        text_prompt = ""
        if request and "messages" in request:
            for msg in request["messages"]:
                if msg["role"] == "user" and "content" in msg:
                    if isinstance(msg["content"], str):
                        text_prompt += msg["content"] + " "
                    elif isinstance(msg["content"], list):
                        for content_item in msg["content"]:
                            if content_item.get("type") == "text":
                                text_prompt += content_item.get("text", "") + " "
        
        return process_vl_request(file_location_or_url, text_prompt)
    
    # Handle regular text-only chat requests
    if isinstance(request, dict):
        # Convert dict to ChatCompletionRequest
        messages = []
        for msg in request.get("messages", []):
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append(ChatMessage(role=msg["role"], content=msg["content"]))
        
        chat_request = ChatCompletionRequest(
            messages=messages,
            max_tokens=request.get("max_tokens", 512),
            temperature=request.get("temperature", 0.2),
            top_p=request.get("top_p", 0.95)
        )
        return generate_chat_response(chat_request)
    
    # If request is already a ChatCompletionRequest object
    return generate_chat_response(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
