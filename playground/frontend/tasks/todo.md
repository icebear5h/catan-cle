# Frontend organization (2026-09-21)

Scope: `playground/frontend` and `evals/catan_board_bench_ui` only.

- [x] Check status, instructions, imports, and baseline tests/builds.
- [x] Move components into cohesive folders, updating all scoped consumers.
- [x] Extract presentational trace sections and state-independent helpers.
- [x] Verify tests/builds, exact built CSS hashes, folder/file counts, and diff.
- [x] Record review and remaining oversized files without claiming compliance.

## Plan

Use direct feature-folder imports, keeping component/CSS pairs together. Avoid
barrel files because dependency traversal can change stylesheet order. Keep CSS
contents and import sequence intact. Split replay transcript presentation and
eval reasoning inspection at existing semantic boundaries; retain state, refs,
effects, and fetching in their original owner components. Keep the larger App,
board renderer, prompt editor, and decision-review state machines out of the
extraction scope.

The activity group uses `activity/`: existing ignore rules exclude `logs/` at
any nesting depth, which would hide moved components from Git.

Alternatives considered: moving only enough files would meet the folder cap but
leave unrelated features mixed; splitting central App state is substantially
riskier than extracting read-only trace presentation.

## Baseline

- Playground: 61 tests pass; TypeScript/Vite build passes (existing chunk warning).
- Eval UI: TypeScript/Vite build passes.
- Playground built CSS SHA-256:
  `dec89013aba4f74e1521c5b0054193bbc190d1f05bb84e331157ce40959034f4`
- Eval built CSS SHA-256:
  `7e4fea163ff0689abe20c7c64557af4f57cbd8d98619975f2adf51dccaadcc4a`

## Review

Completed the 31-file playground component reorganization and four eval file
moves; the exact mapping is in `../README.md`. Added five extracted source
files, all below 300 lines. Playground component folders contain 4/4/6/3/6/10
direct files (activity/board/controls/prompts/replay/traces). The eval source
root contains 13, decisions 2, and reasoning 5. All governed folders within
both frontend roots satisfy the 15-file limit.

The two extraction targets fell from 492 to 279 and 634 to 293 physical lines.
Reviewed the original-to-current diffs: playground App changed only import
paths; eval reasoning's stateful component body is unchanged; replay tab state,
refs, effect dependencies, and narrator markup remain in their original owner.
The extracted model section keeps its original DOM, conditions, text, and keys.
Other moved source files are byte-identical or differ only in import paths.

Validation:

- `npm --prefix playground/frontend test`: 61 passed before and after.
- `npm --prefix playground/frontend run build`: passed; existing >500 kB
  chunk-size warning remains. Rebuilt after renaming the activity folder.
- `npm --prefix evals/catan_board_bench_ui run build`: passed before and after.
- All 15 moved CSS files match their original Git blob hashes. Both complete
  built CSS SHA-256 hashes exactly match the baseline above, preserving cascade
  order as well as stylesheet content. No moved CSS contains relative URLs.
- Scoped `git diff --check`: passed. Updated both existing trace-test imports.
- The repository-mandated full quality gate was also run with `--limit 8`
  (output cap only): exit 1, 210 structure violations, Ruff exit 1, mypy exit 1.
  These unresolved repository-wide failures were reported, not repaired here.

Remaining size debt is 23 source/test files over 300 lines across these roots,
down from 25. In particular, relocated `HexBoard` (756), `PromptSuiteStudio`
(883), `DecisionSpotChecks` (815), `GameControls` (337), `BoardControlsDock`
(317), and oversized stylesheets still exceed the limit. Import-only consumers
`App` (1663), `BenchmarkVerifier` (461), and `BoardFluencyReview` (339) also
remain oversized. They are not counted as compliant extractions.

Two backend docstrings still name the historical flat HexBoard path:
`evals/catan_board_bench/annotations.py` and `render.py`. No runtime imports or
test paths remain at the old frontend locations. Those docstrings are outside
this frontend-only edit scope.

# Continuation: board, controls, and ordered styles

- [x] Inspect the current frontend-only worktree and existing component boundaries.
- [x] Split SftDataExplorer, ReasoningTraces, PlayerInfo, and BoardControlsDock
      styles into ordered, domain-local imports with <=300 lines per file.
- [x] Extract HexBoard geometry/assets and presentational SVG layers; extract
      existing playback actions and the native-reasoning control from controls.
- [x] Run the existing 61 tests and both builds for the completed source state;
      verify unchanged production CSS hashes and all new file/folder limits.
- [x] Record additional closed limits separately from the first pass.

Plan: preserve contiguous CSS rule ranges exactly and keep import entrypoints
at their current paths. Use semantic boundaries, not rule reordering or
selector changes. Keep HexBoard interaction state in HexBoard, with pure
geometry and SVG presentation modules. Existing standalone playback state
can move with its component. Central App orchestration stays outside this pass.

## Continuation review

Seven additional oversized files now meet the limit, including every extracted
implementation file, not only import entrypoints:

| File | Original lines | Current lines | Largest extracted file |
| --- | ---: | ---: | ---: |
| HexBoard.tsx | 756 | 224 | 224 |
| BoardControlsDock.tsx | 317 | 198 | 121 |
| GameControls.tsx | 337 | 292 | 39 |
| PlayerInfo.css | 338 | 2 | 186 |
| BoardControlsDock.css | 302 | 3 | 140 |
| Eval ReasoningTraces.css | 916 | 5 | 253 |
| Eval SftDataExplorer.css | 1005 | 6 | 215 |

The exact source/style mapping and all extracted file sizes are in the README's
continuation section. Board and controls directories now contain 7 and 9 direct
files; their style folders contain 2 and 3. Eval reasoning/styles contains 5,
and sft/styles contains 6. Every governed frontend folder remains <=15 files.
Scoped oversized-file debt is 23 → 16 for this pass (25 → 16 cumulatively).

Review: all CSS rules retain their original sequence, selectors, declarations,
and responsive blocks. HexBoard keeps
coast → terrain/ports → roads → buildings → clickable nodes → annotations draw
order. Its math, asset paths, interaction state, and event handlers are retained.
The old `tile as any` is now the already-narrowed port tile; this changes typing
only. Playback controls retain their seek handlers and remount key. Native
reasoning retains the same options, callback, disabled expression, and markup.

Verification ran once for the completed continuation source state:

- `npm --prefix playground/frontend test`: 61 passed.
- `npm --prefix playground/frontend run build`: passed (existing chunk warning).
- `npm --prefix evals/catan_board_bench_ui run build`: passed.
- Both built CSS SHA-256 hashes match the original baseline above exactly:
  playground `dec89013...959034f4`, eval `7e4fea16...aadcc4a`.
- Scoped line/folder inventory and `git diff --check` pass for the new changes;
  the 16 listed pre-existing oversized files remain debt.

This continuation used scoped structure measurements only; the full repository
gate was not rerun while the other workers were active. The earlier full-gate
failure remains recorded in the first-pass review.
