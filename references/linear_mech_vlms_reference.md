# Linear-Mech-VLMs Reference Code

External repo:

- GitHub: https://github.com/Raphoo/linear-mech-vlms
- Paper: https://arxiv.org/abs/2601.12626
- Local checkout: `references/external/linear-mech-vlms`
- Pinned commit: `1c034d6ecbe1494c450d412f52fdf98e4a4c51c2`

The repo is stored as a git submodule so we can inspect the reference code without
mixing third-party implementation into our Catan modules.

License note: no license file was visible in the checked-out repository. Treat the
code as reference-only unless/until licensing is clarified.

## Relevant Directories

| Directory | Why it matters for Catan |
| --- | --- |
| `spatial_id_derivation/` | Core extraction flow for object-specific spatial IDs, universal spatial IDs, and horizontal/vertical axes. |
| `mirror_attr_swapping/` | Activation swapping between mirrored images and attribute-control images; closest to causal flow evidence. |
| `arbitrary_steering/` | Spatial steering experiments that can inspire atlas-token steering tests. |
| `ground_truth_deviation/` | Spatial ID deviation and image masking diagnostics; useful for failure-mode probes. |
| `spatial_finetuning/` | Auxiliary spatial-ID loss training setup; possible later ablation after basic SFT. |
| `utils/extract_embeds.py` | Embedding extraction utilities to study before building our Qwen3-VL hidden-state capture path. |
| `utils/linalg.py` | Linear algebra helpers for spatial ID directions/projections. |

## Catan Adaptation Sketch

Paper setup:

```text
activation("cup") ~= object_semantics("cup") + spatial_id(current cup location)
```

Catan setup:

```text
activation("<N11>") ~= stable_spatial_id(<N11>)
                    + transient_visual_state(pixels over <N11>)
                    + question/task context
```

Implementation target:

1. Capture hidden states at atlas-token positions such as `<N11>`, `<T09>`,
   `<E23>`, and `<P06>`.
2. Average same-token activations across many fresh boards to isolate stable atlas
   identity.
3. Subtract stable identity from image-conditioned activations to isolate
   transient board state.
4. Probe whether stable identities reconstruct topology and transient residuals
   reconstruct current occupancy/resource/road/port state.
5. Patch activations across contrastive boards to test causal mediation.
