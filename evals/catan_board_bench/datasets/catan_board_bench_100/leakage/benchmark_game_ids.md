# CatanBoardBench Leakage Ledger

Generated: 2026-08-22T00:52:23+00:00

Rule: these game IDs are evaluation-only. Do not use their replay JSON, rendered board images, contracts, QA rows, or derived text in SFT, CPT, validation, prompt tuning, or data selection.

Benchmark: CatanBoardBench-100 with 100 samples and 3032 QA rows, generated at 2026-05-13T21:22:39.101560+00:00.

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

Raw replay IDs currently present locally: 157348754, 178964322, 186086820, 186549889, 187315228, 187949640, 189131938, 189649315, 189933021, 190066582, 191006009, 191010650, 191035308, 191046136, 191057036, 191170335, 191196714, 191208779, 191397826, 191409287, 191493206, 191622286, 191831087, 191834170, 191981351, 192058835, 192193520, 192244603, 192261074, 192281207, 192284486, 192289925, 192294329, 192302493, 192385820, 192418134, 192514349, 192669054, 192754858, 192900367, 192998713, 193008240, 193062324, 193299252, 193379094, 193426507, 193497471, 193538901, 193899417, 193905959, 193960562, 193963991, 194050989, 194134632, 194158047, 194198328, 194209320, 194335024, 194372526, 194507661, 194560883, 194864812, 195196260, 195220811, 195231359, 195313419

Current status: every local raw replay ID overlaps the benchmark held-out set. Training data must come from newly pulled candidate games or another non-overlapping source.

## Candidate Index Inputs

| Index file | Rows |
| --- | --- |
| 4p_games_current.json | 35 |
| 4p_games_me_all.json | 50 |
| 4p_games_top100.json | 8495 |

Wrote 7399 unique training candidate games to `artifacts/raw/colonist/indexes/4p_games_training_candidates.json` after excluding benchmark IDs.
