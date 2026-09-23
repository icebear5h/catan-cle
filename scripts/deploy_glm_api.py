"""
Deploy GLM-4.1V-9B-Thinking as an OpenAI-compatible API on Modal.

Follows the vLLM inference pattern from Modal's examples but adapted for
a HuggingFace vision model (vLLM doesn't support GLM-4.1V's vision encoder).

Usage:
    # Deploy (stays running, gives you a URL)
    modal deploy scripts/deploy_glm_api.py

    # Test locally (spins up ephemeral server)
    modal run scripts/deploy_glm_api.py

    # The deployment serves GET /health, GET /v1/models and
    # POST /v1/chat/completions at
    # https://YOUR-WORKSPACE--glm-4-1v-api-serve.modal.run, in the
    # OpenAI chat-completions shape (string or image_url/text content parts),
    # so any OpenAI client works against .../v1 with any api_key.

Prerequisites:
    pip install modal
    modal setup
    modal secret create huggingface HF_TOKEN=hf_xxx  # optional, for gated models
"""

import base64
import io
import os
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from typing import Protocol

import aiohttp
import modal

# ---------------------------------------------------------------------------
# App & container image
# ---------------------------------------------------------------------------

app = modal.App("glm-4-1v-api")

# GLM-4.1V needs transformers (not vLLM) since vLLM doesn't support its
# vision encoder. We build a container with all deps baked in.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install(
        "torch",
        "torchvision==0.26.0",
        "transformers>=4.46.0,<5.0.0",
        "accelerate",
        "Pillow",
        "sentencepiece",
        "protobuf",
        "fastapi[standard]",
        "uvicorn",
        "httpx",
    )
)

# Only the container has these; Modal skips the block outside the image.
# modal is unannotated, so the context manager goes through a typed handle.
_container_imports: Callable[[], AbstractContextManager[None]] = image.imports

with _container_imports():
    import httpx
    import torch
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from PIL import Image as PILImage
    from transformers import AutoModelForImageTextToText, AutoProcessor


class _Shaped(Protocol):
    """A tokenized tensor, of which only the shape is read here."""
    @property
    def shape(self) -> tuple[int, ...]: ...


class _ChatInputs(Protocol):
    """The processor's tokenized batch, passed on to ``model.generate``."""
    def to(self, device: str) -> "_ChatInputs": ...
    def keys(self) -> Iterable[str]: ...
    def __getitem__(self, key: str) -> _Shaped: ...


class _Processor(Protocol):
    """The unannotated transformers processor, narrowed to what is used here."""
    def apply_chat_template(
        self, conversation: list[dict[str, object]], **kwargs: object
    ) -> _ChatInputs: ...
    def decode(self, token_ids: object, *, skip_special_tokens: bool) -> str: ...


class _ProcessorLoader(Protocol):
    """``AutoProcessor.from_pretrained`` as this script calls it."""
    def __call__(self, model_name: str, /, *, trust_remote_code: bool) -> _Processor: ...


# ---------------------------------------------------------------------------
# Model weights cache (persists across deploys)
# ---------------------------------------------------------------------------

MODEL_NAME = "zai-org/GLM-4.1V-9B-Thinking"
MODEL_REVISION = "main"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

MINUTES = 60
PORT = 8000


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    timeout=10 * MINUTES,
    scaledown_window=5 * MINUTES,  # stay warm for 5 min after last request
)
@modal.concurrent(max_inputs=4)
@modal.web_server(port=PORT, startup_timeout=10 * MINUTES)
def serve() -> None:
    """Start a FastAPI server that exposes an OpenAI-compatible chat completions endpoint."""

    # ---- Load model at startup ----
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

    print(f"Loading {MODEL_NAME} in bf16...")
    load_processor: _ProcessorLoader = AutoProcessor.from_pretrained
    processor = load_processor(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    set_eval_mode: Callable[[], object] = model.eval
    set_eval_mode()
    device = "cuda"
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. VRAM: {vram:.1f} GB")

    # ---- FastAPI app ----
    api = FastAPI(title="GLM-4.1V-9B API", version="1.0")

    @api.get("/health")
    async def health() -> dict[str, str | float]:
        return {"status": "ok", "model": MODEL_NAME, "vram_gb": round(vram, 1)}

    @api.get("/v1/models")
    async def list_models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{
                "id": "glm-4.1v",
                "object": "model",
                "owned_by": "zai-org",
            }],
        }

    @api.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()

        messages = body.get("messages", [])
        temperature = body.get("temperature", 0.2)
        max_tokens = body.get("max_tokens", 4096)

        # Convert OpenAI message format -> GLM format
        glm_messages: list[dict[str, object]] = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if isinstance(content, str):
                glm_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                })
            elif isinstance(content, list):
                glm_content: list[dict[str, object]] = []
                for part in content:
                    if part["type"] == "text":
                        glm_content.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            b64_data = url.split(",", 1)[1]
                            img_bytes = base64.b64decode(b64_data)
                        else:
                            resp = httpx.get(url, timeout=30)
                            img_bytes = resp.content
                        img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                        glm_content.append({"type": "image", "image": img})
                glm_messages.append({"role": role, "content": glm_content})

        inputs = processor.apply_chat_template(
            glm_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(device)

        start = time.time()
        # transformers 5's generate() self type rejects PreTrainedModel under mypy.
        generate = getattr(model, "generate")
        with torch.no_grad():
            outputs = generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        latency_ms = int((time.time() - start) * 1000)

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = processor.decode(generated, skip_special_tokens=True)

        return JSONResponse({
            "id": f"chatcmpl-glm-{int(time.time())}",
            "object": "chat.completion",
            "model": "glm-4.1v",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": int(inputs["input_ids"].shape[1]),
                "completion_tokens": int(len(generated)),
                "total_tokens": int(inputs["input_ids"].shape[1]) + int(len(generated)),
            },
            "latency_ms": latency_ms,
        })

    # ---- Write the app to a temp file and run uvicorn as a subprocess ----
    # @modal.web_server expects the function to RETURN after starting the server,
    # not block. Use subprocess like the vLLM example.
    # Save the FastAPI app so the subprocess can import it
    app_file = "/tmp/_glm_app.py"
    with open(app_file, "w") as f:
        f.write("# Auto-generated -- do not edit\n")
        f.write("import sys; sys.path.insert(0, '/root')\n")

    # We can't easily serialize the model into a subprocess, so instead
    # run uvicorn in-process but in a background thread
    server_thread = threading.Thread(
        target=uvicorn.run,
        args=(api,),
        kwargs={"host": "0.0.0.0", "port": PORT, "log_level": "info"},
        daemon=True,
    )
    server_thread.start()


# ---------------------------------------------------------------------------
# Local test entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
async def test() -> None:
    url = await serve.get_web_url.aio()
    print(f"Server URL: {url}")

    async with aiohttp.ClientSession(base_url=url) as session:
        # Health check
        print("Health check...")
        async with session.get(
            "/health", timeout=aiohttp.ClientTimeout(total=5 * MINUTES)
        ) as resp:
            assert resp.status == 200
            print(f"  OK: {await resp.json()}")

        # Test text-only request
        print("\nText-only test...")
        payload = {
            "model": "glm-4.1v",
            "messages": [
                {"role": "user", "content": "What is 2+2? Answer in one word."},
            ],
            "max_tokens": 32,
        }
        async with session.post("/v1/chat/completions", json=payload) as resp:
            data = await resp.json()
            print(f"  Response: {data['choices'][0]['message']['content']}")
            print(f"  Latency: {data.get('latency_ms')}ms")

    print("\nDeploy with: modal deploy scripts/deploy_glm_api.py")
    print(f"Then use: {url}/v1/chat/completions")
