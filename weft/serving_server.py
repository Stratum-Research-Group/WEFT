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
from typing import List, Optional, Union, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch

from weft import WeftConfig, WeftForCausalLM, WeftCache
from weft.tokenizer import WeftCharTokenizer


app = FastAPI(title="WEFT Inference Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine state
class GlobalEngine:
    model: Optional[WeftForCausalLM] = None
    tokenizer: Optional[WeftCharTokenizer] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: str = "weft-causal-lm"
    paged_manager: Optional[Any] = None

engine = GlobalEngine()


@app.on_event("startup")
def startup_event():
    import os
    model_path = os.environ.get("WEFT_MODEL_PATH")
    if model_path and os.path.exists(model_path):
        load_model(model_path)


def load_model(checkpoint_path: str, device: Optional[str] = None, quantization: Optional[str] = None):
    import os
    if device is not None:
        engine.device = device
    quant = quantization or os.environ.get("WEFT_QUANTIZATION", None)
    engine.model = WeftForCausalLM.from_pretrained(checkpoint_path, quantization=quant).to(engine.device)
    engine.model.eval()

    # Initialize Paged Cache Manager for continuous serving
    try:
        from weft.paged_cache import PagedWeftServingEngine
        latent_dim = getattr(engine.model.config, "mla_latent_dim", 192)
        num_layers = getattr(engine.model.config, "num_hidden_layers", 16)
        engine.paged_manager = PagedWeftServingEngine(
            num_layers=num_layers,
            block_size=16,
            max_blocks=2048,
            latent_dim=latent_dim,
            device=engine.device,
        )
        print(f"[WEFT Server] Initialized PagedWeftServingEngine (2048 blocks, d_latent={latent_dim})")
    except Exception as e:
        print(f"[WEFT Server] Note: Paged cache initialization deferred: {e}")

    vocab_path = os.path.join(checkpoint_path, "vocab.json")
    if os.path.exists(vocab_path):
        engine.tokenizer = WeftCharTokenizer(vocab_file=vocab_path)
    else:
        # Check if model uses BPE (vocab_size 50257) or char tokenizer
        if engine.model.config.vocab_size == 50257:
            import tiktoken
            class TiktokenWrapper:
                def __init__(self):
                    self.enc = tiktoken.get_encoding("gpt2")
                    self.vocab_size = 50257
                def encode(self, text):
                    return self.enc.encode_ordinary(text)
                def decode(self, tokens):
                    return self.enc.decode(tokens)
                def decode_single_token_bytes(self, token_id):
                    return self.enc.decode_single_token_bytes(token_id)
            engine.tokenizer = TiktokenWrapper()
        else:
            if not os.path.exists(vocab_path):
                raise FileNotFoundError(
                    f"Checkpoint at '{checkpoint_path}' does not contain 'vocab.json' "
                    f"required for character-level tokenization (vocab_size={engine.model.config.vocab_size})."
                )
            engine.tokenizer = WeftCharTokenizer(vocab_file=vocab_path)
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
    paged_info = engine.paged_manager.stats() if engine.paged_manager is not None else None
    return {
        "status": "ready" if engine.model is not None else "unloaded",
        "device": engine.device,
        "paged_cache": paged_info,
    }


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
    byte_buffer = bytearray()

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
        eos_id = getattr(engine.model.config, "eos_token_id", None)
        if eos_id is None and hasattr(engine.tokenizer, "enc"):
            eos_id = getattr(engine.tokenizer.enc, "eot_token", 50256)

        is_stop = (token_id == eos_id)
        
        # Byte-safe UTF-8 incremental decode to handle multi-token characters cleanly
        token_char = ""
        if not is_stop:
            if hasattr(engine.tokenizer, "decode_single_token_bytes"):
                byte_buffer.extend(engine.tokenizer.decode_single_token_bytes(token_id))
                try:
                    token_char = byte_buffer.decode("utf-8")
                    byte_buffer.clear()
                except UnicodeDecodeError:
                    token_char = ""
            else:
                token_char = engine.tokenizer.decode([token_id])

        curr_input = next_tok

        finish_reason = "stop" if is_stop else (None if step < max_tokens - 1 else "length")

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
                        "finish_reason": finish_reason,
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
                        "finish_reason": finish_reason,
                    }
                ],
            }

        yield f"data: {json.dumps(chunk)}\n\n"
        if is_stop:
            break

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
    prompt_tokens = len(engine.tokenizer.encode(prompt_text))
    completion_tokens = 0
    final_finish_reason = "length"
    for chunk_str in _generate_tokens_generator(prompt_text, req.max_tokens, req.temperature, req_id, is_chat=True):
        if chunk_str.strip() in ("", "data: [DONE]"):
            continue
        data = json.loads(chunk_str.replace("data: ", ""))
        choice = data["choices"][0]
        content = choice["delta"].get("content", "")
        generated_text += content
        completion_tokens += 1
        if choice.get("finish_reason") is not None:
            final_finish_reason = choice["finish_reason"]

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": engine.model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": final_finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
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
    prompt_tokens = len(engine.tokenizer.encode(prompt_text))
    completion_tokens = 0
    final_finish_reason = "length"
    for chunk_str in _generate_tokens_generator(prompt_text, req.max_tokens, req.temperature, req_id, is_chat=False):
        if chunk_str.strip() in ("", "data: [DONE]"):
            continue
        data = json.loads(chunk_str.replace("data: ", ""))
        choice = data["choices"][0]
        txt = choice.get("text", "")
        generated_text += txt
        completion_tokens += 1
        if choice.get("finish_reason") is not None:
            final_finish_reason = choice["finish_reason"]

    return {
        "id": req_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": engine.model_name,
        "choices": [
            {
                "text": generated_text,
                "index": 0,
                "finish_reason": final_finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# Serve the web playground at root /
import os
playground_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "web_chat")
if os.path.exists(playground_path):
    from fastapi.responses import FileResponse
    @app.get("/")
    def serve_playground():
        return FileResponse(os.path.join(playground_path, "index.html"))
