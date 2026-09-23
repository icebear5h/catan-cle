build_board_fluency_review/{_candidates,_audit,_prompt}.py: DonorView Protocol + generic Candidate[DonorT] so the SFT corpus donor type-checks; prompt takes Mapping; mypy 0->0 ruff 0->0
build_board_fluency_dataset/*: Donor.state JsonDict via as_dict; rows typed JsonDict with meta()/as_* narrowing; ValidationReport/Summary TypedDicts in _sources.py; generic apportion; typed flow graph; mypy 66->0 ruff 0->0
build_spatial_continuation_dataset/*: rows/states/contracts typed JsonDict with as_* narrowing; Pair alias + _pair() for canonical sorted pairs; _select_production return fixed to list[tuple]; metadata files map built via a named JsonDict; mypy 61->0 ruff 0->0
sft/scripts/builders/build_node_factor_dataset/*: AtlasIndices TypedDict + edge_pair/json_objects in _sources; contract/qas/views typed JsonDict with as_* narrowing; atlas_metadata_json(); mypy 33->0 ruff 0->0 (breaks scripts/board_recognition BoardIndices alias, reported to main)
sft/scripts/builders/build_colonist_dummy_fixture.py: JsonLikeDict rows, _pair for edge ids; mypy 7->0 ruff 0->0
sft/scripts/builders/build_atlas_topology_dataset.py: TopologyIndices TypedDict over typed AtlasMetadata, _pair, JsonLikeDict rows; mypy 5->0 ruff 0->0
sft/scripts/builders/build_coordinate_comparison.py: JsonDict rows, json_path/_content narrowing, JsonLikeDict manifest/return; mypy 4->0 ruff 0->0
sft/scripts/builders/build_spatial_continuation_dataset/_outputs.py: score annotated Mapping[str, object] | None; mypy 1->0
sft/scripts/builders/build_board_fluency_review/_build.py: RenderState -> JsonDict via JSON round trip for boards; mypy 1->0
VERIFIED: atlas topology output byte-identical to committed artifact; node_factors full run byte-identical to committed artifacts except pre-existing generator path string
