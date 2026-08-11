"""
Deploy GLM-4.1V-9B-Thinking as an OpenAI-compatible API on Modal.

Follows the vLLM inference pattern from Modal's examples but adapted for
a HuggingFace vision model (vLLM doesn't support GLM-4.1V's vision encoder).

Usage:
    # Deploy (stays running, gives you a URL)
    modal deploy scripts/deploy_glm_api.py

    # Test locally (spins up ephemeral server)
    modal run scripts/deploy_glm_api.py

    # Call the deployed API (OpenAI-compatible):
    curl https://YOUR-WORKSPACE--glm-4-1v-api-serve.modal.run/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "glm-4.1v",
        "messages": [
          {"role": "system", "content": "You are a visual analyzer."},
          {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
            {"type": "text", "text": "What tiles do you see?"}
          ]}
        ]
      }'

    # Or use the openai Python client:
    from openai import OpenAI
    client = OpenAI(base_url="https://YOUR-WORKSPACE--glm-4-1v-api-serve.modal.run/v1", api_key="unused")
    resp = client.chat.completions.create(model="glm-4.1v", messages=[...])

Prerequisites:
    pip install modal
    modal setup
    modal secret create huggingface HF_TOKEN=hf_xxx  # optional, for gated models
"""

import json
import subprocess
import time

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
def serve():
    """Start a FastAPI server that exposes an OpenAI-compatible chat completions endpoint."""

    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    import torch
    import base64
    import io
    import os
    from PIL import Image as PILImage
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # ---- Load model at startup ----
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

    print(f"Loading {MODEL_NAME} in bf16...")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = "cuda"
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. VRAM: {vram:.1f} GB")

    # ---- FastAPI app ----
    api = FastAPI(title="GLM-4.1V-9B API", version="1.0")

    @api.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_NAME, "vram_gb": round(vram, 1)}

    @api.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": "glm-4.1v",
                "object": "model",
                "owned_by": "zai-org",
            }],
        }

    @api.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()

        messages = body.get("messages", [])
        temperature = body.get("temperature", 0.2)
        max_tokens = body.get("max_tokens", 4096)

        # Convert OpenAI message format -> GLM format
        glm_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if isinstance(content, str):
                glm_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                })
            elif isinstance(content, list):
                glm_content = []
                for part in content:
                    if part["type"] == "text":
                        glm_content.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            b64_data = url.split(",", 1)[1]
                            img_bytes = base64.b64decode(b64_data)
                        else:
                            import httpx as hx
                            resp = hx.get(url, timeout=30)
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
        with torch.no_grad():
            outputs = model.generate(
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
    import pickle
    import tempfile
    import subprocess

    # Save the FastAPI app so the subprocess can import it
    app_file = "/tmp/_glm_app.py"
    with open(app_file, "w") as f:
        f.write("# Auto-generated -- do not edit\n")
        f.write("import sys; sys.path.insert(0, '/root')\n")

    # We can't easily serialize the model into a subprocess, so instead
    # run uvicorn in-process but in a background thread
    import threading
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
async def test():
    import aiohttp

    url = await serve.get_web_url.aio()
    print(f"Server URL: {url}")

    async with aiohttp.ClientSession(base_url=url) as session:
        # Health check
        print("Health check...")
        async with session.get("/health", timeout=5 * MINUTES) as resp:
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

    print(f"\nDeploy with: modal deploy scripts/deploy_glm_api.py")
    print(f"Then use: {url}/v1/chat/completions")
