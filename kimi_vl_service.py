# filepath: /Users/hc/working/temp/openapi/kimi_vl_service.py
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from urllib.request import urlopen

model_path = "moonshotai/Kimi-VL-A3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

def process_vl_request(image_path_or_url: str, text_prompt: str):
    try:
        if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
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