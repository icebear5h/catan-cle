# Qwen Catan Modal spending

Provider-metered compute through **September 4, 2026, 1:00 p.m. Pacific** (20:00 UTC). Queried January 1 onward in both configured workspaces; matching runs began May 14. Amounts are USD before credits and exclude shared storage and plan fees.

| Category | Gross compute |
|---|---:|
| Training apps, including in-run evals | $261.71 |
| Standalone evaluation apps | $58.71 |
| Earlier vision probe | $0.01 |
| **Total** | **$320.43** |

The current Qwen3.8 campaign in tetracorp accounts for **$318.99**: $260.90 training and $58.09 standalone evals. Earlier personal-workspace smoke tests, evals, and one probe add $1.44.

## Largest identified experiments

| Experiment | Gross compute |
|---|---:|
| Staged v3 single-piece + tiles | $50.81 |
| Direct v3 mean-init control, 512 steps | $34.45 |
| Direct v3 Gaussian-init treatment, 512 steps | $34.25 |
| pairs_v2 road-heavy continuation, 512 steps | $34.08 |
| pairs_v1 continuation, 256 steps | $19.68 |
| Original marker-only control | $18.60 |
| Single-piece v2 continuation | $18.22 |
| pairs_v2 final regression panel | $13.79 |
| pairs_v1 final regression panel | $8.64 |

Experiments without a confirmed human-readable run label remain attributed to their actual app ID in the complete ledger below.

## Resource costs and live rate

- H200: $244.27 (approximately 53.80 GPU-hours at the current $4.54/hour rate).
- L40S: $1.47.
- CPU: $32.14.
- RAM: $42.55.
- The configured H200 training request is about $6.32/hour including 16 CPU cores and 128 GiB RAM. Recent standalone H200 evals are about $4.62/hour.

## Billing interpretation

- This is actual metered compute, not a GPU-runtime estimate.
- Training app bills include startup, retries, and evaluations performed inside training. They cannot be separated from this billing export.
- The live marker run has only $0.96 included before the cutoff. Later usage is additional; no projected future spend is included.
- Shared volume storage is excluded because the retrieved per-app reports do not allocate it to this project. The tetracorp September workspace snapshot has $36.36 metered volume storage across all projects.
- The tetracorp September billing summary reports $360.108 metered workspace usage, minus $12.288 free storage and $347.82 credits, plus a $250 plan fee, for $250 billed. These are workspace totals with a different collection window; they are not another $250 of Qwen compute. Project-specific net charges cannot be assigned from workspace-wide credits.
- Matching failed/stopped app attempts are included when Modal returned nonzero charges. Zero-cost launches do not appear.
- A separate checkpoint-publishing helper cost $0.00121044 and is outside the SFT/eval/probe total. Other projects and hosted API providers are excluded.

## Complete per-app ledger

| Workspace | App | Confirmed run label or billing description | Category | Gross compute |
|---|---|---|---|---:|
| tetracorp | [ap-J1J9PHV5MyWD3PJZ7jJwqy](https://modal.com/apps/tetracorp/main/ap-J1J9PHV5MyWD3PJZ7jJwqy) | Staged v3 single-piece + tiles | training_including_in_run_evaluation | $50.81 |
| tetracorp | [ap-Ocnfz9zmNomkiQFMls8V90](https://modal.com/apps/tetracorp/main/ap-Ocnfz9zmNomkiQFMls8V90) | Direct v3 mean-init control, 512 steps | training_including_in_run_evaluation | $34.45 |
| tetracorp | [ap-cJCHyYUTNRNuK3jD9H3jTf](https://modal.com/apps/tetracorp/main/ap-cJCHyYUTNRNuK3jD9H3jTf) | Direct v3 Gaussian-init treatment, 512 steps | training_including_in_run_evaluation | $34.25 |
| tetracorp | [ap-5628OGxpUM5CLarPop9j7r](https://modal.com/apps/tetracorp/main/ap-5628OGxpUM5CLarPop9j7r) | pairs_v2 road-heavy continuation, 512 steps | training_including_in_run_evaluation | $34.08 |
| tetracorp | [ap-quF56iRlPhhBGpJyftRifh](https://modal.com/apps/tetracorp/main/ap-quF56iRlPhhBGpJyftRifh) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $24.85 |
| tetracorp | [ap-GJWUfVfWDMWGJ6CGc0nADW](https://modal.com/apps/tetracorp/main/ap-GJWUfVfWDMWGJ6CGc0nADW) | pairs_v1 continuation, 256 steps | training_including_in_run_evaluation | $19.68 |
| tetracorp | [ap-soNGGKHXLygmhCwK3Phb4x](https://modal.com/apps/tetracorp/main/ap-soNGGKHXLygmhCwK3Phb4x) | Original marker-only control | training_including_in_run_evaluation | $18.60 |
| tetracorp | [ap-YT2dybLVHcEM9lLXjPQLzW](https://modal.com/apps/tetracorp/main/ap-YT2dybLVHcEM9lLXjPQLzW) | Single-piece v2 continuation | training_including_in_run_evaluation | $18.22 |
| tetracorp | [ap-iKwx2e32A6pT3lzn851xJV](https://modal.com/apps/tetracorp/main/ap-iKwx2e32A6pT3lzn851xJV) | pairs_v2 final regression panel | standalone_evaluation | $13.79 |
| tetracorp | [ap-spno9s5TpSQ668zloZT3FB](https://modal.com/apps/tetracorp/main/ap-spno9s5TpSQ668zloZT3FB) | pairs_v1 final regression panel | standalone_evaluation | $8.64 |
| tetracorp | [ap-XUiV7Z073WSH4nN2A9YxpL](https://modal.com/apps/tetracorp/main/ap-XUiV7Z073WSH4nN2A9YxpL) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $8.26 |
| tetracorp | [ap-z1FnQguxVsxU9H79lS1eoN](https://modal.com/apps/tetracorp/main/ap-z1FnQguxVsxU9H79lS1eoN) | catan-qwen-series-eval | standalone_evaluation | $5.47 |
| tetracorp | [ap-dw8a95JJBXWvcWaOUEY7MU](https://modal.com/apps/tetracorp/main/ap-dw8a95JJBXWvcWaOUEY7MU) | catan-qwen-series-eval | standalone_evaluation | $3.97 |
| tetracorp | [ap-TTCqSzF68fAtxwbTIUpQdJ](https://modal.com/apps/tetracorp/main/ap-TTCqSzF68fAtxwbTIUpQdJ) | Gaussian diamond-marker run, stopped before replacement | training_including_in_run_evaluation | $3.69 |
| tetracorp | [ap-m3CBdK0eeqfDKbYOMpf52X](https://modal.com/apps/tetracorp/main/ap-m3CBdK0eeqfDKbYOMpf52X) | catan-qwen-series-eval | standalone_evaluation | $3.02 |
| tetracorp | [ap-MjEP6EuStM6eDmuqKzlA7E](https://modal.com/apps/tetracorp/main/ap-MjEP6EuStM6eDmuqKzlA7E) | pairs_v1 occlusion controls | standalone_evaluation | $2.35 |
| tetracorp | [ap-ulF9u7QEXYLGkbIyh4G6Em](https://modal.com/apps/tetracorp/main/ap-ulF9u7QEXYLGkbIyh4G6Em) | Single-piece v4 adjacent-negative run, stopped | training_including_in_run_evaluation | $2.19 |
| tetracorp | [ap-BnjyrZFiDj4U8CeYumhzRm](https://modal.com/apps/tetracorp/main/ap-BnjyrZFiDj4U8CeYumhzRm) | catan-qwen-series-eval | standalone_evaluation | $1.98 |
| tetracorp | [ap-Mnd9K3pqhFnmANsisPmWqQ](https://modal.com/apps/tetracorp/main/ap-Mnd9K3pqhFnmANsisPmWqQ) | Family-aligned row repair evaluation | standalone_evaluation | $1.86 |
| tetracorp | [ap-TBv7wrvSddamDLDRxfbCGd](https://modal.com/apps/tetracorp/main/ap-TBv7wrvSddamDLDRxfbCGd) | Mean-alignment row repair evaluation | standalone_evaluation | $1.85 |
| tetracorp | [ap-P6BD3ptgN6fuCZia9NgYEi](https://modal.com/apps/tetracorp/main/ap-P6BD3ptgN6fuCZia9NgYEi) | catan-qwen-series-eval | standalone_evaluation | $1.84 |
| tetracorp | [ap-j1PEOd0LtSHj3NUe3bY79h](https://modal.com/apps/tetracorp/main/ap-j1PEOd0LtSHj3NUe3bY79h) | catan-qwen-series-eval | standalone_evaluation | $1.79 |
| tetracorp | [ap-NgZjnvQJMF3o0d1N7Gn3CH](https://modal.com/apps/tetracorp/main/ap-NgZjnvQJMF3o0d1N7Gn3CH) | catan-qwen-series-eval | standalone_evaluation | $1.77 |
| tetracorp | [ap-JOTyRLSv9Ap3rPsilCd9pS](https://modal.com/apps/tetracorp/main/ap-JOTyRLSv9Ap3rPsilCd9pS) | catan-qwen-series-eval | standalone_evaluation | $1.74 |
| tetracorp | [ap-FkLohD0umvKzVQ4NyE81KS](https://modal.com/apps/tetracorp/main/ap-FkLohD0umvKzVQ4NyE81KS) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $1.67 |
| tetracorp | [ap-hrm1oVoy2GR31QSPYperoZ](https://modal.com/apps/tetracorp/main/ap-hrm1oVoy2GR31QSPYperoZ) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $1.62 |
| tetracorp | [ap-NdsMnyOE7fPzrckWO2owyD](https://modal.com/apps/tetracorp/main/ap-NdsMnyOE7fPzrckWO2owyD) | catan-qwen-series-eval | standalone_evaluation | $1.41 |
| tetracorp | [ap-0EWsrpD4AGGLg8qtj3CIiF](https://modal.com/apps/tetracorp/main/ap-0EWsrpD4AGGLg8qtj3CIiF) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $1.21 |
| tetracorp | [ap-3zD033fPGb5DwN9VCjjiJm](https://modal.com/apps/tetracorp/main/ap-3zD033fPGb5DwN9VCjjiJm) | Single-piece v6 near/far run, stopped | training_including_in_run_evaluation | $1.19 |
| tetracorp | [ap-VE4UKMjJToiM2FeAAGZojR](https://modal.com/apps/tetracorp/main/ap-VE4UKMjJToiM2FeAAGZojR) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $1.15 |
| tetracorp | [ap-OWGHAochdxkqxzSJXswrCW](https://modal.com/apps/tetracorp/main/ap-OWGHAochdxkqxzSJXswrCW) | catan-qwen-series-eval | standalone_evaluation | $1.06 |
| tetracorp | [ap-K0OaRzUNteoi3apL3WG6EM](https://modal.com/apps/tetracorp/main/ap-K0OaRzUNteoi3apL3WG6EM) | Direct family-word init, stopped and replaced | training_including_in_run_evaluation | $1.05 |
| tetracorp | [ap-vv9OCTqGIbDwNyi0fqZxUd](https://modal.com/apps/tetracorp/main/ap-vv9OCTqGIbDwNyi0fqZxUd) | catan-qwen-series-eval | standalone_evaluation | $1.04 |
| tetracorp | [ap-FxCTlNirGsvjLXJWmHx0w6](https://modal.com/apps/tetracorp/main/ap-FxCTlNirGsvjLXJWmHx0w6) | pairs_v1 checkpoint-64 evaluation | standalone_evaluation | $1.03 |
| tetracorp | [ap-kO4nZVjPpdt6jiPIqzjp8A](https://modal.com/apps/tetracorp/main/ap-kO4nZVjPpdt6jiPIqzjp8A) | Gaussian entity-marker stage 1, active at cutoff | training_including_in_run_evaluation | $0.96 |
| tetracorp | [ap-G7ZApUsiwktgqJra0gEUG8](https://modal.com/apps/tetracorp/main/ap-G7ZApUsiwktgqJra0gEUG8) | catan-qwen-series-eval | standalone_evaluation | $0.66 |
| tetracorp | [ap-uo4HQ2swC1P2LLXzGDThWG](https://modal.com/apps/tetracorp/main/ap-uo4HQ2swC1P2LLXzGDThWG) | catan-qwen-series-eval | standalone_evaluation | $0.50 |
| tetracorp | [ap-N8QB13iaQFIQUqdZrWl22a](https://modal.com/apps/tetracorp/main/ap-N8QB13iaQFIQUqdZrWl22a) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $0.38 |
| tetracorp | [ap-SmkyFaQNTmi5UKJQd46ZJm](https://modal.com/apps/tetracorp/main/ap-SmkyFaQNTmi5UKJQd46ZJm) | Single-piece v4 initial full-epoch launch, stopped | training_including_in_run_evaluation | $0.36 |
| tetracorp | [ap-ajARKI5Brc1VxyygHQn4d5](https://modal.com/apps/tetracorp/main/ap-ajARKI5Brc1VxyygHQn4d5) | Qwen3.8 training smoke | training_including_in_run_evaluation | $0.36 |
| tetracorp | [ap-xy0bhh0Mhoc2kj3M8NoyII](https://modal.com/apps/tetracorp/main/ap-xy0bhh0Mhoc2kj3M8NoyII) | Qwen3.8 spatial/robber curriculum smoke | training_including_in_run_evaluation | $0.36 |
| tetracorp | [ap-LOxYQx13vXR8ND3TigrtPu](https://modal.com/apps/tetracorp/main/ap-LOxYQx13vXR8ND3TigrtPu) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $0.34 |
| tetracorp | [ap-2s2ZTyucK2aNBvJjtOgvcQ](https://modal.com/apps/tetracorp/main/ap-2s2ZTyucK2aNBvJjtOgvcQ) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $0.29 |
| tetracorp | [ap-y0m8wdtJLKpmlh5c55AF5l](https://modal.com/apps/tetracorp/main/ap-y0m8wdtJLKpmlh5c55AF5l) | catan-qwen-series-eval | standalone_evaluation | $0.25 |
| tetracorp | [ap-AuE4sQgeHstcUPKdBuGRLW](https://modal.com/apps/tetracorp/main/ap-AuE4sQgeHstcUPKdBuGRLW) | Single-piece v5 negative mix, stopped | training_including_in_run_evaluation | $0.25 |
| icebear5h | [ap-A33q3RQkm6xjJnN4jdiqgY](https://modal.com/apps/icebear5h/main/ap-A33q3RQkm6xjJnN4jdiqgY) | catan-qwen-series-eval | standalone_evaluation | $0.22 |
| tetracorp | [ap-ZdajtkpjnDqKCweCSbl1Lf](https://modal.com/apps/tetracorp/main/ap-ZdajtkpjnDqKCweCSbl1Lf) | catan-qwen-series-eval | standalone_evaluation | $0.21 |
| tetracorp | [ap-HBuk2vM3cY5BW4qExXZK5l](https://modal.com/apps/tetracorp/main/ap-HBuk2vM3cY5BW4qExXZK5l) | catan-qwen-series-eval | standalone_evaluation | $0.20 |
| tetracorp | [ap-6m4b7cqyqwqiuCGLqRSI4F](https://modal.com/apps/tetracorp/main/ap-6m4b7cqyqwqiuCGLqRSI4F) | catan-qwen-series-eval | standalone_evaluation | $0.20 |
| tetracorp | [ap-x9NYU99LeyvoavNFJidmL2](https://modal.com/apps/tetracorp/main/ap-x9NYU99LeyvoavNFJidmL2) | catan-qwen-series-eval | standalone_evaluation | $0.19 |
| tetracorp | [ap-BRhvqQxEskZVXoGjuBQPBX](https://modal.com/apps/tetracorp/main/ap-BRhvqQxEskZVXoGjuBQPBX) | catan-qwen-series-eval | standalone_evaluation | $0.19 |
| tetracorp | [ap-vSp0co3ShMXqzmfUAsZQyB](https://modal.com/apps/tetracorp/main/ap-vSp0co3ShMXqzmfUAsZQyB) | catan-qwen-series-eval | standalone_evaluation | $0.18 |
| tetracorp | [ap-hwCIewObaadMYmDVX9idc9](https://modal.com/apps/tetracorp/main/ap-hwCIewObaadMYmDVX9idc9) | catan-qwen-series-eval | standalone_evaluation | $0.18 |
| tetracorp | [ap-PrHITwfndxpYoUw2KfHocH](https://modal.com/apps/tetracorp/main/ap-PrHITwfndxpYoUw2KfHocH) | catan-qwen-series-eval | standalone_evaluation | $0.18 |
| icebear5h | [ap-isKUVsdqZK3PBIXqGE6Y4h](https://modal.com/apps/icebear5h/main/ap-isKUVsdqZK3PBIXqGE6Y4h) | catan-qwen-series-sft | training_including_in_run_evaluation | $0.17 |
| tetracorp | [ap-65Cy4JOiCq7HScZAxtRDIV](https://modal.com/apps/tetracorp/main/ap-65Cy4JOiCq7HScZAxtRDIV) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $0.17 |
| tetracorp | [ap-oTDi1txPrmaj1M988HxziS](https://modal.com/apps/tetracorp/main/ap-oTDi1txPrmaj1M988HxziS) | catan-qwen-series-eval | standalone_evaluation | $0.16 |
| tetracorp | [ap-QZh8xAuICfBBWNNA5jCVvR](https://modal.com/apps/tetracorp/main/ap-QZh8xAuICfBBWNNA5jCVvR) | catan-qwen-series-eval | standalone_evaluation | $0.16 |
| icebear5h | [ap-OkbKhqvjK2BwVk87ZlFDzB](https://modal.com/apps/icebear5h/main/ap-OkbKhqvjK2BwVk87ZlFDzB) | catan-qwen-series-eval | standalone_evaluation | $0.15 |
| icebear5h | [ap-OHJ13rYv2jzQC8RSSwy5eQ](https://modal.com/apps/icebear5h/main/ap-OHJ13rYv2jzQC8RSSwy5eQ) | catan-qwen-series-eval | standalone_evaluation | $0.14 |
| icebear5h | [ap-ou0Qa0sC7wNlK4PDBlDqFg](https://modal.com/apps/icebear5h/main/ap-ou0Qa0sC7wNlK4PDBlDqFg) | catan-qwen-series-sft | training_including_in_run_evaluation | $0.13 |
| tetracorp | [ap-PyqZqW7od2LYeju1z3Ihj3](https://modal.com/apps/tetracorp/main/ap-PyqZqW7od2LYeju1z3Ihj3) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $0.13 |
| tetracorp | [ap-areZO9nMgvxBXKMWBW87yn](https://modal.com/apps/tetracorp/main/ap-areZO9nMgvxBXKMWBW87yn) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $0.12 |
| icebear5h | [ap-mUBWUVdYaO8BZU9NqkYBYQ](https://modal.com/apps/icebear5h/main/ap-mUBWUVdYaO8BZU9NqkYBYQ) | catan-qwen-series-eval | standalone_evaluation | $0.12 |
| tetracorp | [ap-FXjZROd0E8Z2Gc0i71cMQF](https://modal.com/apps/tetracorp/main/ap-FXjZROd0E8Z2Gc0i71cMQF) | catan-qwen-series-eval | standalone_evaluation | $0.11 |
| icebear5h | [ap-R48z9ojREzfuVNAvMMfofM](https://modal.com/apps/icebear5h/main/ap-R48z9ojREzfuVNAvMMfofM) | Historical Qwen3.5-9B one-step smoke | training_including_in_run_evaluation | $0.09 |
| icebear5h | [ap-QenaP8rcPVtT8ecRYnvtZ8](https://modal.com/apps/icebear5h/main/ap-QenaP8rcPVtT8ecRYnvtZ8) | catan-qwen-vl-sft | training_including_in_run_evaluation | $0.09 |
| icebear5h | [ap-CG4RUN6kAUGIJoiBdM7eqG](https://modal.com/apps/icebear5h/main/ap-CG4RUN6kAUGIJoiBdM7eqG) | catan-qwen-vl-sft | training_including_in_run_evaluation | $0.09 |
| tetracorp | [ap-tmYqhYfq5GpxK8lqmQnnYS](https://modal.com/apps/tetracorp/main/ap-tmYqhYfq5GpxK8lqmQnnYS) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $0.08 |
| tetracorp | [ap-YyCwN4c8Ww5OPXSwZJYFdc](https://modal.com/apps/tetracorp/main/ap-YyCwN4c8Ww5OPXSwZJYFdc) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $0.07 |
| icebear5h | [ap-n3j5XuK39zIXYlE0zGoO9e](https://modal.com/apps/icebear5h/main/ap-n3j5XuK39zIXYlE0zGoO9e) | catan-qwen-series-sft | training_including_in_run_evaluation | $0.07 |
| tetracorp | [ap-obdTNJNfhBY43jAp16DxXz](https://modal.com/apps/tetracorp/main/ap-obdTNJNfhBY43jAp16DxXz) | catan-qwen-series-eval | standalone_evaluation | $0.06 |
| icebear5h | [ap-3dGxbAF0nPnxKANLTUQJ9W](https://modal.com/apps/icebear5h/main/ap-3dGxbAF0nPnxKANLTUQJ9W) | Historical Qwen3-VL-8B smoke | training_including_in_run_evaluation | $0.05 |
| tetracorp | [ap-ljq31CcYAXzVEkn15ytFjM](https://modal.com/apps/tetracorp/main/ap-ljq31CcYAXzVEkn15ytFjM) | catan-qwen-series-eval | standalone_evaluation | $0.05 |
| tetracorp | [ap-9edLReNUDIdLmUFVLR5y2g](https://modal.com/apps/tetracorp/main/ap-9edLReNUDIdLmUFVLR5y2g) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $0.05 |
| icebear5h | [ap-xg8pkC78r9YEFsPLuXTBiO](https://modal.com/apps/icebear5h/main/ap-xg8pkC78r9YEFsPLuXTBiO) | catan-qwen-vl-sft | training_including_in_run_evaluation | $0.04 |
| icebear5h | [ap-Yd50o4HOztjNo46IXMkZ5b](https://modal.com/apps/icebear5h/main/ap-Yd50o4HOztjNo46IXMkZ5b) | Historical Qwen3.5-9B ten-step smoke | training_including_in_run_evaluation | $0.03 |
| icebear5h | [ap-qdaGRGysBycMWKtsioaCNn](https://modal.com/apps/icebear5h/main/ap-qdaGRGysBycMWKtsioaCNn) | catan-qwen-series-sft | training_including_in_run_evaluation | $0.02 |
| icebear5h | [ap-Nn5wl0iPwlmI9xS8N0X9nC](https://modal.com/apps/icebear5h/main/ap-Nn5wl0iPwlmI9xS8N0X9nC) | catan-qwen-series-sft | training_including_in_run_evaluation | $0.02 |
| icebear5h | [ap-kfvdT1hUmoXT6TC0lJeD58](https://modal.com/apps/icebear5h/main/ap-kfvdT1hUmoXT6TC0lJeD58) | Historical Qwen3.8 vision probe | vision_probe | $0.01 |
| icebear5h | [ap-nWLvFUdAZwqG7LCXAZkvk8](https://modal.com/apps/icebear5h/main/ap-nWLvFUdAZwqG7LCXAZkvk8) | catan-qwen-series-sft | training_including_in_run_evaluation | $0.01 |
| tetracorp | [ap-ENue3ejaCIJf4MhD2SIyb8](https://modal.com/apps/tetracorp/main/ap-ENue3ejaCIJf4MhD2SIyb8) | catan-qwen-series-eval | standalone_evaluation | $0.01 |
| icebear5h | [ap-Le0IuEu4Pefvs4dJpHCr6b](https://modal.com/apps/icebear5h/main/ap-Le0IuEu4Pefvs4dJpHCr6b) | catan-qwen-series-sft | training_including_in_run_evaluation | $0.00 |
| tetracorp | [ap-zWi2iDIzyyT2SOYV61Ekfl](https://modal.com/apps/tetracorp/main/ap-zWi2iDIzyyT2SOYV61Ekfl) | catan-qwen3-8-vision-sft | training_including_in_run_evaluation | $0.00 |
| icebear5h | [ap-iqHYStiKXzUWPySEK92bos](https://modal.com/apps/icebear5h/main/ap-iqHYStiKXzUWPySEK92bos) | catan-qwen-series-sft | training_including_in_run_evaluation | $0.00 |
| icebear5h | [ap-i6Pderg5FsK90qgVCKMJaq](https://modal.com/apps/icebear5h/main/ap-i6Pderg5FsK90qgVCKMJaq) | catan-qwen-vl-sft | training_including_in_run_evaluation | $0.00 |
| icebear5h | [ap-rW2zbT8nSJkX0lFluGeqht](https://modal.com/apps/icebear5h/main/ap-rW2zbT8nSJkX0lFluGeqht) | catan-qwen-series-sft | training_including_in_run_evaluation | $0.00 |
| icebear5h | [ap-YfKNuCBvLI7ncNZWnKcsbY](https://modal.com/apps/icebear5h/main/ap-YfKNuCBvLI7ncNZWnKcsbY) | catan-qwen-series-sft | training_including_in_run_evaluation | $0.00 |

Billing source: [Modal billing API/CLI](https://modal.com/docs/cli/latest/billing). [Billing documentation](https://modal.com/docs/guide/billing) explains pre-credit reporting and possible collection delays.

Exact eight-decimal costs, original resource/interval rows, scope, and retrieval timestamps are preserved in [the JSON ledger](./2026-09-04-modal-spending.json).
