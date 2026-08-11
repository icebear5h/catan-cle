# CatanBench Leakage Ledger

Generated: 2026-05-14T00:34:16+00:00

Rule: these game IDs are evaluation-only. Do not use their replay JSON, rendered board images, contracts, QA rows, or derived text in SFT, CPT, validation, prompt tuning, or data selection.

Benchmark: CatanBench-100 with 100 samples and 3032 QA rows, generated at 2026-05-13T21:22:39.101560+00:00.

## Held-Out Benchmark Games

| Game ID | Samples | Replay file | Replay steps |
| --- | --- | --- | --- |
| 191035308 | 8 | 191035308.json, data_pipeline/bootstrapping/data/raw_replays/191035308.json | 8, 78, 149, 219, 290, 360, 431, 501 |
| 191046136 | 8 | 191046136.json, data_pipeline/bootstrapping/data/raw_replays/191046136.json | 8, 79, 149, 220, 291, 362, 432, 503 |
| 191057036 | 8 | 191057036.json, data_pipeline/bootstrapping/data/raw_replays/191057036.json | 8, 78, 149, 219, 290, 360, 431, 501 |
| 191196714 | 8 | 191196714.json, data_pipeline/bootstrapping/data/raw_replays/191196714.json | 8, 89, 169, 250, 330, 411, 491, 572 |
| 191208779 | 8 | 191208779.json, data_pipeline/bootstrapping/data/raw_replays/191208779.json | 8, 82, 156, 230, 305, 379, 453, 527 |
| 192244603 | 8 | 192244603.json, data_pipeline/bootstrapping/data/raw_replays/192244603.json | 8, 61, 114, 167, 219, 272, 325, 378 |
| 192261074 | 8 | 192261074.json, data_pipeline/bootstrapping/data/raw_replays/192261074.json | 8, 68, 128, 188, 247, 307, 367, 427 |
| 192281207 | 8 | 192281207.json, data_pipeline/bootstrapping/data/raw_replays/192281207.json | 8, 68, 127, 187, 246, 306, 365, 425 |
| 192284486 | 8 | 192284486.json, data_pipeline/bootstrapping/data/raw_replays/192284486.json | 8, 85, 161, 238, 314, 391, 467, 544 |
| 192289925 | 8 | 192289925.json, data_pipeline/bootstrapping/data/raw_replays/192289925.json | 8, 61, 114, 167, 219, 272, 325, 378 |
| 192294329 | 8 | 192294329.json, data_pipeline/bootstrapping/data/raw_replays/192294329.json | 8, 78, 148, 218, 287, 357, 427, 497 |
| 192418134 | 8 | 192418134_sample.json, data_pipeline/bootstrapping/data/raw_replays/192418134_sample.json | 8, 9, 10, 11, 13, 14, 15, 16 |
| 194335024 | 4 | 194335024.json, data_pipeline/bootstrapping/data/raw_replays/194335024.json | 8, 60, 112, 164 |

## Local Replay Inventory

Raw replay IDs currently present locally: 191035308, 191046136, 191057036, 191196714, 191208779, 192244603, 192261074, 192281207, 192284486, 192289925, 192294329, 192418134, 194335024

Current status: every local raw replay ID overlaps the benchmark held-out set. Training data must come from newly pulled candidate games or another non-overlapping source.

## Candidate Index Inputs

| Index file | Rows |
| --- | --- |
| 4p_games_current.json | 35 |
| 4p_games_me_all.json | 50 |
| 4p_games_top100.json | 8495 |

Wrote 7399 unique training candidate games to `data_pipeline/bootstrapping/scrapers/4p_games_training_candidates.json` after excluding benchmark IDs.
