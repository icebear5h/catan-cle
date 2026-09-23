# Stock Qwen3.8-27B: exact Cartesian evaluation

Run `miles-cartesian-stock-20260921-r01` completed on **2026-09-21 at 17:15:52 UTC**.
Stock Qwen3.8-27B answered **143/200 correctly (71.5%)**, without fine-tuning,
adapters, added tokenizer tokens, or enabled thinking. All raw responses were
downloaded and independently rescored against the pinned source cases.

## Results

| Operation | Correct | Accuracy |
| --- | ---: | ---: |
| Direction yes/no | 25/32 | 78.1% |
| Direction choice | 30/32 | 93.8% |
| Node/tile neighbors | 7/32 | 21.9% |
| Entity incidence | 3/16 | 18.8% |
| Piece owner | 31/32 | 96.9% |
| Owned nodes/buildings | 21/24 | 87.5% |
| Owned roads | 16/16 | 100% |
| Owned roads incident to a node | 10/16 | 62.5% |
| **Total** | **143/200** | **71.5%** |

Grouped results: direction **55/64**, adjacency **7/32**, incidence **3/16**,
ownership **78/88**. No answers hit the output limit. Thirteen predictions failed
answer admission: **12 referenced unknown Cartesian positions** and one failed
the required atom syntax. These are included as incorrect, not dropped.

### Pure readout control

The ownership family includes 16 **owned-incident-road** questions, which require
a geometric join. Excluding those, direct supplied-state readouts score **68/72
(94.4%)**: piece owner 31/32, owned nodes 21/24, owned roads 16/16.

Auditing the actual visible prompts confirms four genuine errors:

- BLACK buildings: both `N(0,-1) black settlement` and
  `N(-sqrt(3),2) black settlement` were supplied; the model returned only the first.
- BLUE buildings: a blue settlement and a blue city were supplied; the model
  returned only the settlement.
- BLACK settlements: it selected `N(sqrt(3),-2) black city` instead of the supplied
  `N(-sqrt(3),2) black settlement`.
- Piece owner: `N(-sqrt(3)/2,-5/2) orange settlement` was explicitly supplied;
  the model answered `NONE`.

All four outputs were syntactically admissible, and none were truncated. Their
prompts were 2,508–2,522 tokens. The evidence establishes omissions, a piece-type
selection error, and a lookup miss; it does not isolate whether dense input
serialization, coordinate notation, prompt wording, or decoding causes them.

### Token-budget audit

The user correctly challenged the input size. This run used the **dense 155-record
full-board readout**, not the sparse minimal representation. The original stock
tokenizer was retrieved and its SHA-256 matched to preflight; all 200 recorded
prompt-token totals reproduce exactly.

For the first BLACK-building miss (2,508 tokens), token attribution is:

| Component | Tokens |
| --- | ---: |
| Board coordinate addresses | 1,811 |
| Other board values/punctuation | 392 |
| Geometry/output legend | 257 |
| Question + participants + chat wrapper | 48 |

That board explicitly enumerated **109 empty node/edge slots**. Omitting empty
slots and stating `Unlisted nodes and edges are empty.` retains 46 records.
The sparse board is **174 noncoordinate tokens + 500 coordinate tokens + 9 tokens
for the empty-default rule = 683 tokens**. Retaining the same legend/question/
wrappers gives **988 tokens**, down from 2,508. Across all 88 dynamic questions,
this count-only sparse projection averages 1,058.5 versus 2,514.6 tokens.

Coordinate attribution assigns boundary-spanning tokens to coordinates when they
overlap an address; coarse sections are attributed by token start. These counts
are reproducible in `token_budget_audit.json`. The sparse projection was **only
tokenized, not evaluated**; the 143/200 result remains a dense-input baseline and
does not establish performance on the intended minimal input.

Further count-only comparison on the same 88 sparse dynamic boards, using the
verified stock tokenizer (board plus empty-default rule; h definition included):

| Coordinate spelling | Example | Mean board tokens |
| --- | --- | ---: |
| ASCII radical | `N(sqrt(3)/2,1/2)` | 746.4 |
| Unicode radical | `N(√3/2,1/2)` | 707.7 |
| `h = sqrt(3)/4` constant | `N(2h,0.5)` | 611.3 |
| Three-decimal approximation | `N(0.866,0.5)` | 740.1 |
| Six-decimal approximation | `N(0.866025,0.5)` | 876.8 |

The h spelling saves about 18% of sparse-board tokens without rounding. Decimal
spelling is not automatically cheaper for this tokenizer. No model inference
was performed for these variants; counts are retained in
`coordinate_spelling_token_audit.json` and do not establish accuracy differences.

All seven direction yes/no misses answered `no` when the gold was `yes`.
The run does not isolate the cause of those mistakes. Topology failures include
extra/missing neighbors, wrong signs/offsets, and nonexistent positions; they are
not merely formatting failures.

## Interpretation

Stock Qwen reads most supplied ownership and makes most coordinate comparisons,
but the pure readout control is not yet error-free. Enumerating correct hex
neighbors and incidence sets remains unreliable under greedy, no-thinking generation.

This is a **Cartesian-only stock-model screen**, not a matched result against
atlas IDs or the trained r04 checkpoint. Its 71.5% is not directly comparable to
the earlier board-fluency SFT panels. The same 200 source cases were retained:
198 test and two validation port-incidence cases, with 61 distinct dynamic states
across 88 ownership questions. The static board is fixed; this is not evidence
of gameplay strength or novel-topology generalization.

## Conditions and identity

- Model: `Qwen/Qwen3.8-27B`, revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Miles: `12754e9507e64d5e537288da17793246e913c525`.
- Official amd64 image:
  `radixark/miles@sha256:6628bff749ffd32e6a62b479a1128daee25a8c0e3c28eb86301a0d620f5dd598`.
- One NVIDIA H200, BF16, no quantization, max concurrency 16, CUDA graphs disabled.
- Greedy: temperature 0, top-k 1, top-p 1; `enable_thinking=false`.
- Context 4,096; output allowance 512; one answer per case; no row filtering.
- Longest actual stock-tokenizer prompt: 2,544 tokens; longest gold: 80 tokens.
- Prepared panel SHA-256:
  `6a6ab54375e17cc2ac39ab9a99baf28da2411d7bcfc05269507ce3c3ff76e1b3`.
- Input source SHA-256:
  `ee2f3bb8824881dcf0740f8267b7cbfc577a11b57540e2755379345b51a86c64`.
- Actual input/output tokens: 457,429 / 2,986; means 2,287.145 / 14.93;
  longest generated response 80 tokens.

## Runtime, cost, and workspace

CPU preflight took 104.46 function seconds. The H200 function took 379.92 seconds,
including Miles/Ray/SGLang initialization; the logged 200-case evaluation loop
took approximately 41 seconds. The first official-image import separately took
about ten minutes; the local command timed out before functions were created.
Its empty app was stopped and the cached image reused. Exactly one GPU eval ran.

At Modal's September 21 published rates, the recorded CPU/GPU function windows
estimate **$0.65**. This excludes container startup/termination, image building,
storage, and subscriptions; it is **not the provider's actual bill**.

The completed run belongs to workspace **`icebear5h`**:
[app `ap-c299dzp6nqmgY5ce4W4hsN`](https://modal.com/apps/icebear5h/main/ap-c299dzp6nqmgY5ce4W4hsN).
It was explicitly stopped at 17:16:13 UTC and verified at zero tasks. The user
subsequently selected **`tetracorp` as the default Modal profile**; historical run
ownership has not changed.

## Artifacts and launch notes

Local root: `artifacts/runs/sft/miles-cartesian-stock-20260921-r01/`.

- `analysis.json`: independently verified scores, per-task/family counts, all
  incorrect answers, token totals, file hashes, and compute-window estimate.
- `remote/evaluation/results/eval-0/cartesian/records.jsonl`: all 200 raw answers.
- `remote/evaluation/results/eval-0/`: per-panel and overall summaries.
- `remote/evaluation/debug/eval_0.pt`: original Miles sample dump.
- `remote/preflight.json`, `remote/gpu.json`, `remote/worker.log`: exact runtime,
  checkpoint hashes, commands, timings, and full logs.

`sft/miles_eval/modal_run.py` supplies the bounded CPU-preflight/H200 boundary.
The official image uses `/opt/sglang/bin/python`; additional packages must be
installed into that interpreter rather than assuming `uv pip --system` selects
it. This launch installed Modal/Catan dependencies there before preflight.
