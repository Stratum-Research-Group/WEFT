"""
Live Local Inference Server for WEFT Studio.
Provides genuine, dynamic autoregressive chat completions via physical local weights.
Exposes OpenAI-compatible /v1/chat/completions endpoint on port 8089.
"""

import os
import glob
import time
import json
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

app = FastAPI(title="WEFT Live Local Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HUB = os.path.expanduser("~/.cache/huggingface/hub")
# Use Qwen2.5-0.5B-Instruct for conversational alignment
instruct_paths = glob.glob(f"{HUB}/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*")
if instruct_paths:
    model_path = instruct_paths[0]
else:
    model_path = glob.glob(f"{HUB}/models--Qwen--Qwen2.5-Coder-0.5B/snapshots/*")[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Loading live model weights from: {model_path} onto {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
).to(device).eval()
print("[+] Live local model initialized and ready for chat.")

@app.get("/health")
def health():
    return {"status": "healthy", "engine": "weft-live-local-inference", "model": "Qwen2.5-Coder-0.5B"}

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    data = await req.json()
    messages = data.get("messages", [])
    model_id = data.get("model", "qwen2.5-coder-0.5b")
    stream = data.get("stream", False)
    
    # Format chat prompt
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = ""
        for m in messages:
            r = m.get("role", "user")
            c = m.get("content", "")
            if r == "system":
                prompt += f"<|im_start|>system\n{c}<|im_end|>\n"
            elif r == "user":
                prompt += f"<|im_start|>user\n{c}<|im_end|>\n"
            elif r == "assistant":
                prompt += f"<|im_start|>assistant\n{c}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Dynamic generation parameters
    max_new_tokens = data.get("max_tokens", 256)
    temperature = max(float(data.get("temperature", 0.7)), 0.1)

    with torch.no_grad():
        im_end_id = tokenizer.encode("<|im_end|>")[0]
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=im_end_id,
            eos_token_id=[im_end_id, tokenizer.eos_token_id]
        )

    # Extract new assistant tokens only
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    # Strip any trailing repeat loops or stop tokens
    for stop_str in ["<|im_end|>", "<|endoftext|>", "\norative", "\n\norative"]:
        if stop_str in response_text:
            response_text = response_text.split(stop_str)[0].strip()

    if stream:
        async def event_generator():
            words = response_text.split(" ")
            for i, w in enumerate(words):
                chunk = {
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"delta": {"content": w + (" " if i < len(words)-1 else "")}, "index": 0, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8089)
