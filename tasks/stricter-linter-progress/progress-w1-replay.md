# Wave 1 - cle/replay
baseline: ruff 247, mypy 369, pytest 4 failed / 441 passed / 33 skipped (tests/replay tests/audits tests/viewer)
cle/replay/contracts.py (NEW, 96) - shared protocols/aliases - ruff+mypy clean
cle/replay/runtime/access.py (25) - annotated - ruff+mypy clean
cle/replay/runtime/revision.py (11) - annotated - ruff+mypy clean
cle/replay/colonist/types.py (NEW, 38) - Colonist archive TypedDicts - clean
cle/replay/colonist/constants.py (110) - Final annotations - clean
cle/replay/colonist/coordinates.py (126) - annotated - clean
cle/replay/colonist/types.py (118) - +ActionHint/TradeLedgerRecord/ResourceMismatch
cle/replay/colonist/helpers.py (154) - annotated - clean
cle/replay/colonist/event_parser.py (596) -> package: __init__ 265, trades.py 213, game_log.py 195, resources.py 124 - ruff+mypy clean
  verified: 53 real archives / 24992 parsed rows byte-identical old vs new, incl. all 1866 print lines
cle/replay/runtime/audit.py (249) -> package: __init__ 84, final_state.py 197, issues.py 55, state.py 17 - ruff+mypy clean
cle/replay/runtime/checkpoint.py (85) - annotated, Any removed - clean
cle/replay/runtime/trade_ledger.py (116) - annotated - clean
mid-point: ruff 247->148, mypy 369->246; remaining: activity, action_matcher, navigation, step_executor
cle/replay/runtime/action_matcher.py (388) -> package: __init__ 103, trades.py 195, plays.py 164, context.py 124, builds.py 82 - clean
cle/replay/runtime/navigation.py (309) -> package: __init__ 88, goto.py 231, undo.py 77 - clean
cle/replay/runtime/step_executor.py (1773) -> package (14 files, max 206 lines):
  __init__ 92, context 110, offers 206, publishing 165, forcing 256, trade_steps 266
  direct/: __init__ 51, outcome 57, trades 89, dev_cards 163, robber 117, builds 179
  step/: __init__ 117, prepare 142, unmatched 208, skip_reason 88, execute 178, logging 113
cle/replay/activity.py (208) - Any removed - clean
ALL of cle/replay: ruff 0, mypy 0, structure 0
FIX: hint_list narrowed to isinstance(list) broke tests/viewer/commentary duck-typed parsed_actions;
     replaced with a runtime_checkable ParsedActions protocol (__len__/__getitem__). Back to baseline.
FIX: ReplayRuntimeState.replay_mutation_lock made a read-only property so threading.RLock satisfies it.
VERIFIED: 53 real Colonist archives x up to 400 steps, HEAD vs new worktree, PYTHONHASHSEED=0:
     26.8MB behavioural digest byte-identical (steps, actions, events, game_log, issues, ledger,
     player_state, roads, robber, divergence, undo) and 36842 stdout lines byte-identical.
FINAL: ruff 0, mypy 0, structure 0, git diff --check clean,
       pytest 4 failed / 441 passed / 33 skipped == baseline.
