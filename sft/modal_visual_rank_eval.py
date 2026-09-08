"""One capped H200 call: intact v2 versus original+rank-r visual matrix deltas."""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import modal

from sft.modal_qwen_series_eval import eval_image, hf_cache, sft_data, sft_runs
from sft.scripts.extract_visual_delta import sha256_file, visual_keys, write_json
from sft.visual_rank_eval import paired_summary, reconstruct_weight, select_terrain_boards


app = modal.App("catan-visual-rank-mini-eval")
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SNAPSHOT = "/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/" + REVISION
ADAPTER = "/runs/catan-vision-sft/catan-qwen38-gauss-s2-terrain-20260904/398f0a023ec9/checkpoints/checkpoint-384"
FACTORS = "/runs/catan-vision-diagnostics/visual-delta-gauss-s2-ck384-20260905-fp64/derived"
DATA = "/data/catan-vision-sft/datasets/398f0a023ec9"
OUTPUT = "/runs/qwen-series-eval/visual-rank-mini-20260906"
VARIANTS = (("intact_v2", None), ("rank256", 256), ("rank64", 64),
            ("rank16", 16), ("rank8", 8), ("rank0_matrices", 0))
LOCAL_VALIDATION = "artifacts/generated/board_recognition/replay_v1/terrain_readout_v1/stage1/validation.jsonl"


@app.function(
    image=eval_image, gpu="H200", cpu=(8.0, 8.0), memory=(65536, 65536),
    timeout=900, startup_timeout=300, retries=0, max_containers=1, scaledown_window=2,
    volumes={"/cache": hf_cache, "/data": sft_data, "/runs": sft_runs},
)
def evaluate(expected_content_sha256: str) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from sft.scripts.eval_qwen_vl_adapter import image_reference, iter_jsonl, load_model, run_eval_job

    started = time.monotonic()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    output = Path(OUTPUT)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    result = {"schema": "catan_visual_rank_mini/v1", "status": "preflight", "variants": [],
              "adapter": ADAPTER, "factors": FACTORS, "hf_revision": REVISION,
              "scope": "all 112 visual matrices; biases/norms, language LoRA and atlas rows remain at v2",
              "rank0_note": "original visual matrices with v2 vectors/language/tokens; NOT original Qwen",
              "precision": "visual FP32 master weights, BF16 CUDA autocast; all variants identical",
              "gpu": "H200", "timeout_seconds": 900, "retries": 0}
    write_json(output / "sweep.json", result)
    try:
        rows = [row for _, row in iter_jsonl(Path(DATA) / "eval.jsonl")]
        chosen, manifest = select_terrain_boards(rows)
        if (manifest["source_rows"], manifest["rows"], manifest["layouts"]) != (3072, 240, 5):
            raise ValueError("unexpected terrain validation coverage")
        if manifest["content_sha256"] != expected_content_sha256:
            raise ValueError("remote prompts/targets/selection differ from local preflight")
        image_hashes = {}
        for row in chosen:
            reference = Path(image_reference(row))
            candidates = [reference] if reference.is_absolute() else [Path(DATA) / reference, Path(DATA) / "images" / reference]
            image_path = next((path for path in candidates if path.is_file()), None)
            if image_path is None:
                raise FileNotFoundError(reference)
            row["image"] = str(image_path)
            row.pop("images", None)
            if str(image_path) not in image_hashes:
                image_hashes[str(image_path)] = sha256_file(image_path)
        manifest["image_sha256"] = image_hashes
        write_json(output / "selection.json", manifest)
        selected_path = output / "selected.jsonl"
        selected_path.write_text("".join(json.dumps(row) + "\n" for row in chosen))

        report = json.loads((Path(FACTORS) / "report.json").read_text())
        if report["status"] != "complete" or report["hf_revision"] != REVISION:
            raise ValueError("wrong or incomplete SVD report")
        if sha256_file(Path(ADAPTER) / "visual_model.safetensors") != report["trained_sha256"]:
            raise ValueError("v2 visual checkpoint hash mismatch")
        for shard in report["base_shards"]:
            if sha256_file(Path(SNAPSHOT) / shard["file"]) != shard["sha256"]:
                raise ValueError("base shard hash mismatch")
        index = json.loads((Path(SNAPSHOT) / "model.safetensors.index.json").read_text())["weight_map"]
        base = {}
        for shard in report["base_shards"]:
            with safe_open(str(Path(SNAPSHOT) / shard["file"]), framework="pt") as source:
                for row in report["tensors"]:
                    if index[row["base_key"]] == shard["file"]:
                        base[row["name"]] = source.get_tensor(row["base_key"]).bfloat16().float()
        trained_raw = load_file(Path(ADAPTER) / "visual_model.safetensors")
        trained = {row["name"]: trained_raw[row["trained_key"]] for row in report["tensors"]}
        factors = {}
        for row in report["tensors"]:
            if "factor_file" not in row:
                continue
            path = Path(FACTORS) / row["factor_file"]
            if sha256_file(path) != row["factor_sha256"]:
                raise ValueError(f"SVD factor hash mismatch: {path}")
            with safe_open(str(path), framework="pt") as source:
                factors[row["name"]] = {key: source.get_tensor(key) for key in ("lora_A", "lora_B")}
        if len(base) != 333 or len(factors) != 112 or any(t.dtype != torch.float32 for t in trained.values()):
            raise ValueError("unexpected factor/weight coverage or source precision")
        result.update(status="loading_model", selection_sha256=manifest["content_sha256"],
                      trained_visual_sha256=report["trained_sha256"], matrices=len(factors))
        write_json(output / "sweep.json", result)
        print(json.dumps({"event": "preflight_passed", "rows": len(chosen), "matrices": len(factors)}), flush=True)
        with (output / "model_load.log").open("w") as log, contextlib.redirect_stdout(log):
            model, processor, evidence = load_model(
                model_id=SNAPSHOT, adapter_dir=ADAPTER, bits=16,
                disable_flash_attn2=True, token_inventory=f"{DATA}/trainable_tokens.json",
            )
        parameters = dict(model.named_parameters())
        names = visual_keys(list(parameters))
        if set(names) != set(trained):
            raise ValueError("runtime visual parameter set does not match source")
        with torch.no_grad():
            for key, name in names.items():
                parameter = parameters[name]
                if parameter.device.type != "cuda":
                    raise ValueError("unexpected offloaded visual parameter")
                parameter.data = parameter.data.float()
                parameter.copy_(trained[key])
                if not torch.equal(parameter.cpu(), trained[key]):
                    raise ValueError("v2 FP32 restore failed")
        model.requires_grad_(False)
        model.eval()
        args = argparse.Namespace(
            image_root=None, limit=None, batch_size=48, long_batch_size=5,
            max_new_tokens=16, long_max_new_tokens=512, candidate_scoring=False,
            occlusion_margin=0.03, model_id=SNAPSHOT, adapter_dir=ADAPTER,
            bits=16, token_inventory=f"{DATA}/trainable_tokens.json",
        )
        evidence["rank_sweep_precision"] = result["precision"]
        for label, rank in VARIANTS:
            if time.monotonic() - started > 820:
                raise TimeoutError("not starting another variant near the 15-minute limit")
            result.update(status="running", current_variant=label)
            write_json(output / "sweep.json", result)
            sft_runs.commit()
            print(json.dumps({"event": "variant_start", "variant": label,
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
            variant_started = time.monotonic()
            with torch.no_grad():
                for key, name in names.items():
                    # CPU FP32 reconstruction avoids autocast/TF32 changing the SVD product.
                    value = reconstruct_weight(base[key], trained[key], factors.get(key), rank)
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(f"nonfinite reconstruction: {key}")
                    parameters[name].copy_(value)
            torch.cuda.synchronize()
            directory = output / label
            directory.mkdir()
            with (directory / "eval.log").open("w") as log, contextlib.redirect_stdout(log):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    run_eval_job(model=model, processor=processor, adapter_evidence=evidence,
                                 args=args, eval_jsonl=str(selected_path), image_variant="original",
                                 output_dir=directory)
            records = [row for _, row in iter_jsonl(directory / "records.jsonl")]
            metrics = paired_summary(records, manifest["row_ids"])
            (directory / "strict_records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
            compact = {"label": label, "rank": rank, **metrics,
                       "elapsed_seconds": time.monotonic() - variant_started,
                       "matrix_energy_captured": 1.0 if rank is None else (
                           0.0 if rank == 0 else report["summary"]["all"]["matrix_energy_weighted_rank_capture"][str(rank)])}
            write_json(directory / "paired_summary.json", compact)
            result["variants"].append(compact)
            write_json(output / "sweep.json", result)
            sft_runs.commit()
            print(json.dumps({"event": "variant_complete", "label": label,
                              "heads": metrics["heads"], "readouts": metrics["readouts"],
                              "seconds": compact["elapsed_seconds"]}), flush=True)
            if rank is None and any(head["accuracy"] < 0.95 for head in metrics["heads"].values()):
                raise RuntimeError("intact v2 control unexpectedly weak; stopping the sweep for audit")
        with torch.no_grad():
            for key, name in names.items():
                parameters[name].copy_(trained[key])
                if not torch.equal(parameters[name].cpu(), trained[key]):
                    raise ValueError("post-sweep v2 restore failed")
        result.update(status="complete", exact_v2_restore_verified=True)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "sweep.json", result)
        sft_runs.commit()
    return result


@app.local_entrypoint()
def main(dry_run: bool = True):
    from sft.scripts.eval_qwen_vl_adapter import iter_jsonl

    _, manifest = select_terrain_boards([row for _, row in iter_jsonl(Path(LOCAL_VALIDATION))])
    if (manifest["source_rows"], manifest["rows"], manifest["layouts"]) != (3072, 240, 5):
        raise ValueError("unexpected local dataset coverage")
    plan = {"dry_run": dry_run, "output": OUTPUT, "gpu": "H200", "timeout_seconds": 900,
            "retries": 0, "variants": [label for label, _ in VARIANTS], "selection": manifest,
            "maximum_900s_resource_cost_at_20260906_list_rates": 900 * (0.001261 + 8 * 0.0000131 + 64 * 0.00000222),
            "cost_exclusions": "image build, startup, storage, teardown and any infrastructure restart"}
    print(json.dumps({**plan, "selection": {key: value for key, value in manifest.items() if key != "row_ids"}}, indent=2))
    if dry_run:
        return
    result = evaluate.remote(manifest["content_sha256"])
    print(json.dumps({"status": result["status"], "output": OUTPUT,
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
