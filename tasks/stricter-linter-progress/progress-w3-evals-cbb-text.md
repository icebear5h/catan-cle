evals/catan_board_bench/text_representations/schema.py: BoardFacts/BoardFact{Tile,Node,Edge,Port} TypedDicts; public_board_facts takes Mapping[str, JsonValue] and narrows contract rows (as_dicts/as_str/as_int); JsonDict alias now re-exports evals.json_types.JsonDict; 115-contract render/parse/digest witness byte-identical; mypy 1->0
evals/catan_board_bench/text_representations/{__init__,json_formats,dsl_format,ascii_format}.py: signatures on BoardFacts; BoardFacts exported from the package; witness identical; mypy 0 ruff 0
scripts/board_bench/run/eval_catan_board_bench_text_formats/run.py: facts_by_sample typed dict[str, BoardFacts]; mypy 1->0
evals/catan_board_bench/scoring/categories.py: JsonDict alias now re-exports evals.json_types.JsonDict (swapped after the rest of the family was typed); mypy 1->0
evals/catan_board_bench/scoring/normalization.py: target params Mapping[str, JsonValue]; contains_value/contains_*count take object (they only str() it); find_by_token token_value: object; mypy 0 ruff 0
evals/catan_board_bench/scoring/answers.py: target-scored categories dispatch to _score_target(category, as_dict(target), normalized); hex/fallback paths untouched; token lists/ratios narrowed with as_list/as_str/as_dicts; mypy ~140->0
evals/catan_board_bench/scoring/prompting.py: contract/target/category narrowed (as_dict/as_dicts/as_str); coord ints via _coord_at; token joins via _tokens; mypy ~104->0
evals/catan_board_bench/scoring/selection.py: qas typed list[JsonDict]; sample_id/category/contract_path narrowed with as_str; mypy ~91->0
evals/catan_board_bench/scoring/summaries.py: model_key/category/score/latency narrowed; per-model summary built then stored (same dict); mypy ~16->0
[verify] scoring witness over 6288 QA rows (4 question sets, 4 selection modes, prompts with/without atlas, local atlas, sentinel, 81744 score_answer calls, summarize/summarize_records) byte-identical; text_representations 115-contract witness byte-identical
