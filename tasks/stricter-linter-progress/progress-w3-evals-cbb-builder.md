evals/catan_board_bench/builder/shapes.py (new): rows/member/text/texts/whole_number/json_list narrowing helpers over evals.json_types; mypy 0 ruff 0
evals/catan_board_bench/builder/selection.py: honest JsonDict, rows()/as_list/as_int reads, _find_by_id takes JsonValue id, disconnected pair built as (low, high); mypy 0->0 ruff 0->0
evals/catan_board_bench/builder/questions.py: AddQuestion target typed evals.json_types.JsonDict; mypy 0->0
evals/catan_board_bench/builder/questions_board.py: rows/member/text reads, occupied-node targets via json_list; mypy 0->0
evals/catan_board_bench/builder/questions_graph.py: rows/text/texts reads, json_list targets; mypy 0->0
evals/catan_board_bench/builder/questions_players.py: rows/text/texts reads, node_pair/edge_node_pair typed JsonValue, token lists via json_list; mypy 0->0
evals/catan_board_bench/builder/suite.py: contract collections typed JsonList, int lists via json_list, qa_pairs reads sample/robber/tiles via as_dict/member/rows/whole_number; mypy 0->0
evals/catan_board_bench/builder/{records,replay}.py: import honest JsonDict; mypy 0->0
evals/catan_board_bench/builder/benchmark_builder.py: source/sample_meta/manifest_record/metadata annotated JsonDict, replay game_id narrowed to str|None, counters dict built by comprehension (same content); mypy 0->0
evals/catan_board_bench/builder/constants.py: JsonDict = Dict[str, Any] replaced by re-export of evals.json_types.JsonDict; mypy 1->0
VERIFY: qa/questions/answer_key JSONL regenerated from all 100 committed contracts byte-identical to before AND to committed dataset files; fresh public_board_contract + qa_pairs from 4 replays x 4 steps (both sample-index parities) byte-identical before/after; 3-sample build_catan_board_bench_sync smoke OK; all touched modules import.
