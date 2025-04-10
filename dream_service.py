# filepath: /Users/hc/working/temp/openapi/dream_service.py
import time
from fastapi import HTTPException
from transformers import AutoModel, AutoTokenizer
import torch
from typing import List, Optional
from pydantic import BaseModel

model = None
tokenizer = None

def initialize_model(model_path: str):
    global model, tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.to(device).eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    choices: List[ChatCompletionChoice]

def generate_chat_response(request):
    if isinstance(request, dict):
        request = ChatCompletionRequest(**request)
        
    try:
        # Prepare input
        inputs = tokenizer.apply_chat_template(
            [msg.dict() for msg in request.messages],
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True
        )
        input_ids = inputs.input_ids.to(device=model.device)
        attention_mask = inputs.attention_mask.to(device=model.device)

        # Generate response
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=request.max_tokens,
            output_history=True,
            return_dict_in_generate=True,
            steps=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            alg="entropy",
            alg_temp=0.,
        )

        # Decode response
        generations = [
            tokenizer.decode(g[len(p):].tolist())
            for p, g in zip(input_ids, output.sequences)
        ]
        content = generations[0].split(tokenizer.eos_token)[0]
        
        # Prepare response
        response_message = ChatMessage(role="assistant", content=content)
        choice = ChatCompletionChoice(
            index=0,
            message=response_message,
            finish_reason="length" if len(content) >= request.max_tokens else "stop"
        )

        return ChatCompletionResponse(
            id="chatcmpl-" + str(hash(content)),
            created=int(time.time()),
            choices=[choice]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))