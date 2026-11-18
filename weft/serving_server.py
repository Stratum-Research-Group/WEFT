"""
High-Throughput OpenAI-Compatible Streaming Serving Server for WEFT.
Implements:
1. /v1/chat/completions (streaming and non-streaming)
2. /v1/completions (streaming and non-streaming)
3. /v1/models (catalog)
4. /health (readiness check)
"""

import json
import time
import uuid
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import torch

from weft import WeftConfig, WeftForCausalLM, WeftCache
from weft.tokenizer import WeftCharTokenizer


app = FastAPI(title="WEFT Inference Server", version="1.0.0")

# Global engine state
class GlobalEngine:
    model: Optional[WeftForCausalLM] = None
    tokenizer: Optional[WeftCharTokenizer] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: str = "weft-causal-lm"

engine = GlobalEngine()


def load_model(checkpoint_path: str, device: Optional[str] = None):
    import os
    if device is not None:
        engine.device = device
    engine.model = WeftForCausalLM.from_pretrained(checkpoint_path).to(engine.device)
    engine.model.eval()
    vocab_path = os.path.join(checkpoint_path, "vocab.json")
    if os.path.exists(vocab_path):
        engine.tokenizer = WeftCharTokenizer(vocab_file=vocab_path)
    else:
        engine.tokenizer = WeftCharTokenizer()
    print(f"[WEFT Server] Loaded model from {checkpoint_path} on {engine.device}")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "weft-causal-lm"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "weft-causal-lm"
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    stream: bool = False


@app.get("/health")
def health():
    return {"status": "ready" if engine.model is not None else "unloaded", "device": engine.device}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": engine.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "weft",
            }
        ],
    }


def _generate_tokens_generator(prompt_text: str, max_tokens: int, temperature: float, req_id: str, is_chat: bool):
    if engine.model is None or engine.tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    device = engine.device
    raw_tokens = engine.tokenizer.encode(prompt_text)
    if isinstance(raw_tokens, list):
        tokens = torch.tensor([raw_tokens], dtype=torch.long, device=device)
    else:
        tokens = raw_tokens.unsqueeze(0).to(device)
    b, s = tokens.shape

    past_key_values = None
    curr_input = tokens
    created = int(time.time())

    for step in range(max_tokens):
        with torch.no_grad():
            outputs = engine.model(input_ids=curr_input, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)

        token_id = next_tok[0].item()
        token_char = engine.tokenizer.decode([token_id])
        curr_input = next_tok

        if is_chat:
            chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": engine.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token_char},
                        "finish_reason": None if step < max_tokens - 1 else "length",
                    }
                ],
            }
        else:
            chunk = {
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": engine.model_name,
                "choices": [
                    {
                        "text": token_char,
                        "index": 0,
                        "finish_reason": None if step < max_tokens - 1 else "length",
                    }
                ],
            }

        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if engine.model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    prompt_text = "".join([f"{m.role}: {m.content}\n" for m in req.messages]) + "assistant: "
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if req.stream:
        return StreamingResponse(
            _generate_tokens_generator(prompt_text, req.max_tokens, req.temperature, req_id, is_chat=True),
            media_type="text/event-stream",
        )

    # Non-streaming
    generated_text = ""
    for chunk_str in _generate_tokens_generator(prompt_text, req.max_tokens, req.temperature, req_id, is_chat=True):
        if chunk_str.strip() in ("", "data: [DONE]"):
            continue
        data = json.loads(chunk_str.replace("data: ", ""))
        generated_text += data["choices"][0]["delta"].get("content", "")

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": engine.model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_text),
            "completion_tokens": len(generated_text),
            "total_tokens": len(prompt_text) + len(generated_text),
        },
    }


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if engine.model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    prompt_text = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    req_id = f"cmpl-{uuid.uuid4().hex[:12]}"

    if req.stream:
        return StreamingResponse(
            _generate_tokens_generator(prompt_text, req.max_tokens, req.temperature, req_id, is_chat=False),
            media_type="text/event-stream",
        )

    generated_text = ""
    for chunk_str in _generate_tokens_generator(prompt_text, req.max_tokens, req.temperature, req_id, is_chat=False):
        if chunk_str.strip() in ("", "data: [DONE]"):
            continue
        data = json.loads(chunk_str.replace("data: ", ""))
        generated_text += data["choices"][0].get("text", "")

    return {
        "id": req_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": engine.model_name,
        "choices": [
            {
                "text": generated_text,
                "index": 0,
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_text),
            "completion_tokens": len(generated_text),
            "total_tokens": len(prompt_text) + len(generated_text),
        },
    }
