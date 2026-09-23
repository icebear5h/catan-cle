sft/scripts/eval/failure_scorecard/_rows.py: ReadoutSkips TypedDict, error_rate helper, as_* narrowing in user_prompt/synthetic_board/edge_orientations; mypy 22->0 ruff 0->0
sft/scripts/eval/failure_scorecard/_scorer.py: TokenStats dataclass, typed Counter/defaultdict[list[int]]/readout dict[str,int] state, JSON built at end (recall_json/readout_json); mypy 273->0 ruff 0->0
sft/scripts/eval/failure_scorecard/_sets.py: load_json_dict/loads_json + as_* narrowing, typed found list; mypy 12->0 ruff 0->0
sft/scripts/eval/failure_scorecard/_report.py: as_dict/opt_dict narrowing, as_number guard for table, typed scored/set_entry locals; mypy 77->0 ruff 0->0
sft/scripts/eval/failure_scorecard/_base.py: import sort (ruff I001); mypy 0 ruff 1->0
