# Original Qwen → terrain v2 visual delta

Completed 2026-09-05. Extracted the full visual-tower/merger weight difference between the original Hugging Face checkpoint and the good terrain **checkpoint-384**, then decomposed each matrix with SVD. This is an analysis artifact, not an installed adapter or a training launch.

## Result

The original→v2 visual update is not well reproduced by a rank-16 approximation: rank 16 per matrix captures **51.33% of its squared weight-change norm**. The exact v2 weights remain untouched.

| Rank per matrix | Tower energy captured | Merger energy captured | Combined matrix energy captured |
|---:|---:|---:|---:|
| 8 | 42.08% | 38.10% | 38.38% |
| 16 | 51.45% | 51.32% | 51.33% |
| 32 | 61.45% | 64.96% | 64.71% |
| 64 | 71.19% | 77.28% | 76.86% |
| 128 | 80.04% | 86.98% | 86.49% |
| 256 | 87.81% | 93.58% | 93.18% |
| 512 | 94.55% | 97.31% | 97.11% |

Energy is the squared Frobenius norm of the parameter difference, **not accuracy or retained knowledge**. Combined percentages weight matrices by their delta energy; they are not averages of layer percentages. The 110 linear weights alone give essentially the same result: 51.325% at rank 16.

This does **not** determine the rank needed for a new piece-learning adapter. The rank used to describe/protect an old update and the rank used to learn a new update are separate choices. No compressed-model accuracy evaluation was run.

## Sources and precision

Base model: `Qwen/Qwen3.8-27B`, HF revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Target on Modal Volume `catan-sft-runs`:

```text
catan-vision-sft/catan-qwen38-gauss-s2-terrain-20260904/398f0a023ec9/checkpoints/checkpoint-384/visual_model.safetensors
```

The original run config did not pin an HF revision. The selected revision was the sole snapshot in the shared training cache, agreed with its `refs/main`, and was verified against the HF revision API. This is cache-backed lineage evidence, not a revision recorded in the original run config.

All 333 visual tensors reside in the original model's first shard. The cached shard's SHA256 matches the revision API's LFS metadata; downloading the entire language model was unnecessary.

| Artifact | SHA256 |
|---|---|
| HF `model-00001-of-00018.safetensors` | `ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c` |
| v2 FP32 visual checkpoint | `94534c119727435bc2011fb76a6038c1e71facd120abbf482e71bd80a2d252f2` |
| Exact dense delta | `711c6bffe2b7dd1b55faec2ba4b52ec96047be5ed19801cd98eac345fe5b3ef3` |
| Rank-16 linear factors | `9dff0339e8deb54d0ced0af3a05f523fab901cbc369d925644d5f82476f1ae73` |

The trainer loads the base in BF16 and promotes the visual master weights to FP32. Extraction therefore uses the same baseline:

```text
base_as_loaded = HF_weight.to(bfloat16).to(float32)
dense_delta = v2_weight.to(float64) - base_as_loaded.to(float64)
reconstructed = (base_as_loaded.to(float64) + dense_delta).to(float32)
```

All **333 tensors / 460,730,096 parameters** were FP32 in the target checkpoint, matched the original visual key set and shapes, were finite, and passed bit-for-bit reconstruction. FP32 subtraction initially failed the exact round-trip guard through cancellation; the final dense delta uses FP64. Exactness was checked on every actual tensor, not assumed from the dtype. SVD is separately computed in FP32.

This difference includes the accumulated marker-stage and terrain-stage updates, optimizer effects, and any intermediate export rounding. It is **not** an isolated terrain gradient or a direct measurement of forgetting.

## Factor coverage and convention

- 110 linear weight matrices: tower attention/MLP and merger, with a compact rank-16 export.
- 1 positional embedding matrix and 1 flattened patch-convolution kernel: SVD diagnostics saved, explicitly **not** ordinary linear-LoRA targets.
- Biases and normalization vectors: preserved in the exact dense delta, not silently discarded or represented as LoRA factors.
- Language LoRA and atlas-token rows: outside this extraction's scope.

For each matrix, the full singular spectrum and top-256 input/output bases are saved. With `D = U S Vᵀ`, the factor convention is:

```text
B = U[:, :r] * sqrt(S[:r])
A = sqrt(S[:r])[:, None] * Vt[:r, :]
D_approx = B @ A                 # scaling multiplier = 1
```

For smaller ranks, slice `B[:, :r]` and `A[:r, :]`. These files are **not a drop-in PEFT adapter**: module targeting, scaling, omitted non-linear tensors, and serialization/loading still require explicit integration. The rank-16 export contains 220 FP32 A/B tensors and is 32,038,712 bytes.

Keep the exact v2 visual weights as the frozen starting point if testing a new adapter. The exported SVD bases can describe candidate protected subspaces without compressing the old checkpoint. Weight-change magnitude alone does not identify retention-sensitive directions; orthogonality to these bases is not a guarantee of preserved behavior. Do not add the old delta again on top of v2.

## Artifacts

Local compact results, under the repository's ignored SFT diagnostics directory:

- [Machine-readable report](../../artifacts/diagnostics/sft/visual_delta_gauss_s2_ck384_20260905/derived/report.json): every tensor, singular spectrum, source mapping, source/output hashes, and reconstruction checks.
- [Per-matrix CSV](../../artifacts/diagnostics/sft/visual_delta_gauss_s2_ck384_20260905/derived/matrix_spectra.csv).
- [Rank-16 linear A/B factors](../../artifacts/diagnostics/sft/visual_delta_gauss_s2_ck384_20260905/derived/visual_delta_rank16_linear.safetensors).
- [Provenance](../../artifacts/diagnostics/sft/visual_delta_gauss_s2_ck384_20260905/source/provenance.json) and [CPU job receipt](../../artifacts/diagnostics/sft/visual_delta_gauss_s2_ck384_20260905/cpu_receipt.json).

The full dense FP64 delta and all top-256 bases/factors remain on **Modal Volume `catan-sft-runs`**, not downloaded locally:

```text
catan-vision-diagnostics/visual-delta-gauss-s2-ck384-20260905-fp64/derived/
  visual_delta.safetensors
  visual_delta_rank16_linear.safetensors
  svd_factors/<canonical-weight-name>.safetensors
  report.json
  matrix_spectra.csv
  README.md
```

Successful app: `ap-vzRiUMvAAm9sk66eklhQ9G`. CPU-only: 8 CPUs, 24,576 MiB memory limit, 1,200-second timeout, no retries. Core extraction/SVD took **89.72 seconds**; that excludes image build, startup, source hashing before extraction, and transfers. This is not a billing estimate. No GPU, training, adapter installation, or behavioral evaluation was performed.

## Implementation and verification

- [Extractor](../../sft/scripts/extract_visual_delta.py): source/hash/key/dtype validation; exact dense difference; per-matrix SVD, bases, factors, spectra and reports; refuses to overwrite an existing output directory.
- [Bounded CPU launcher](../../sft/modal_visual_delta.py): uses the cached original and v2 checkpoint; writes only a new diagnostics directory. Default entrypoint is a dry run. Its completed destination intentionally cannot be reused without choosing a new destination.
- [Tests](../../tests/test_extract_visual_delta.py): precision, cancellation, source validation, factor orientation, non-LoRA tensor handling, and source-preserving end-to-end extraction.

Focused validation: **36 tests passed** across the extractor, token-row inspector, and trainer test files. Local checks additionally verified the downloaded factor-file hash, all 220 tensor shapes/finiteness, and rank-16 factor energy against the recorded singular spectra.

Numerical checks across the actual extraction:

- Maximum SVD energy relative discrepancy: `9.01e-7`.
- Maximum saved input/output basis orthogonality errors: `4.65e-6` / `2.50e-6`.
- Maximum locally checked rank-16 factor-energy relative discrepancy: `3.61e-6`.
- No saved basis exceeded its matrix's measured FP32 numerical rank.

The run-review guidance shaped the checkpoint-lineage, component-scope, and FP32 checks. Existing model/checkpoint files were not modified. This report supplies a candidate basis for the proposed constrained-update experiment; no retention claim or next training launch is implied.
