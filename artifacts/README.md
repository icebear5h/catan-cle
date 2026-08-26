# Repository artifacts

This tree stores data and run evidence separately from installable Python
packages.

## Layout

- `raw/colonist/replays/`: canonical captured Colonist replay payloads.
- `raw/colonist/indexes/`: leaderboard/history candidate indexes.
- `staging/colonist/replays/`: unverified and rejected replay captures.
- `manifests/colonist/splits/`: deterministic split and queue manifests.
- `fixtures/catan_board_bench/smoke5/`: frozen five-game visual fixture.
- `fixtures/sft/`: tracked renderer and infrastructure smoke fixtures.
- `generated/pretraining/legacy_corpus/`: historical corpus outputs.
- `generated/sft/`: ignored, regenerable SFT datasets.
- `generated/catan_board_bench/piece_recognition/`: ignored piece-probe inputs.
- `diagnostics/sft/`: ignored local renderer/tokenizer diagnostics.
- `runs/catan_board_bench/`: model-evaluation plans, responses, and summaries.
- `runs/sft/`: local checkpoints/logs and compact accepted SFT evidence.

Raw payloads may contain third-party usernames and user IDs. Do not publish or
copy them outside approved storage without reviewing identity and rights
metadata. Never edit a raw payload in place; derive a new versioned artifact.

Generated runs are append-only evidence. A valid run records its inputs, hashes,
settings, model/provider, costs or usage, and deterministic scoring output.
Transient retries and partial files should be quarantined rather than silently
merged into an accepted run.

The historical artifacts moved here remain versioned to preserve the existing
research record. The content-preservation receipt is
`artifacts/manifests/layout_migrations/data_pipeline_layout_v1.json`. The SFT
cleanup receipt is `artifacts/manifests/layout_migrations/sft_layout_v1.json`.
The CatanBoardBench namespace receipt is
`artifacts/manifests/layout_migrations/catan_board_bench_rename_v1.json`.
Verify them without network access using
`python scripts/verify_data_pipeline_layout.py`,
`python scripts/verify_sft_layout.py`, and
`python scripts/verify_catan_board_bench_rename.py`. New large corpora should
use an explicit external-storage or LFS policy before they are added to Git.
