# CatanBoardBench-100 OpenRouter Baseline

Scope: stored OpenRouter eval runs under `artifacts/runs/catan_board_bench/catan_board_bench_100/openrouter`. These are the frozen baseline numbers before any Catan-specific tuning.

## Overall Runs

| Run | Model | Requests | Attempted | Exact | Component | Categories |
| --- | --- | --- | --- | --- | --- | --- |
| 20260513T200550Z | gemma3-4b-free | 12 | 0 | 0.0% | 0.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T200550Z | nemotron-12b-free | 12 | 12 | 50.0% | 65.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T200550Z | qwen3-vl-8b | 12 | 12 | 41.7% | 55.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T200550Z | ui-tars-7b | 12 | 12 | 41.7% | 60.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T200750Z | gemma3-4b-free | 3 | 0 | 0.0% | 0.0% | robber_tile, tile_resource_number, node_occupancy |
| 20260513T200855Z | gemma3-12b | 6 | 6 | 50.0% | 60.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T200855Z | gemma3-4b | 6 | 6 | 0.0% | 10.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T200855Z | ministral-3b | 6 | 6 | 50.0% | 40.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T200855Z | ministral-8b | 6 | 6 | 33.3% | 50.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T200855Z | nemotron-3-nano-free | 6 | 6 | 0.0% | 10.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T200855Z | qwen3.5-9b | 6 | 6 | 0.0% | 0.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T200941Z | gemma4-26b-a4b-free | 6 | 0 | 0.0% | 0.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T200941Z | gemma4-31b-free | 6 | 0 | 0.0% | 0.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T200941Z | llama-3.2-11b | 6 | 6 | 0.0% | 10.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T200941Z | mistral-small-3.2-24b | 6 | 6 | 33.3% | 40.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T200941Z | qwen3-vl-8b-thinking | 6 | 6 | 33.3% | 60.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T201207Z | gemma3-12b | 30 | 30 | 33.3% | 49.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T201207Z | ministral-3b | 30 | 30 | 30.0% | 37.3% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T201207Z | nemotron-12b-free | 30 | 20 | 35.0% | 48.5% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T201207Z | qwen3-vl-8b | 30 | 30 | 43.3% | 51.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T201207Z | ui-tars-7b | 30 | 30 | 33.3% | 51.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260513T212010Z | qwen3-vl-8b | 40 | 40 | 20.0% | 14.7% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, port_trade_type, color_building_counts, color_road_count |
| 20260513T212255Z | qwen3-vl-8b | 40 | 40 | 7.5% | 8.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, port_trade_type, color_building_counts, color_road_count |
| 20260513T212456Z | google_gemini-3.1-pro-preview | 40 | 40 | 7.5% | 12.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, port_trade_type, color_building_counts, color_road_count |
| 20260514T222811Z_qwen3_vs_qwen3.5 | qwen3-vl-8b | 65 | 0 | 0.0% | 0.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260514T222811Z_qwen3_vs_qwen3.5 | qwen3.5-9b | 65 | 0 | 0.0% | 0.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260514T222854Z_qwen3_vs_qwen3.5 | qwen3-vl-8b | 30 | 30 | 3.3% | 25.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260514T222854Z_qwen3_vs_qwen3.5 | qwen3.5-9b | 30 | 30 | 0.0% | 0.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260514T225626Z_qwen3.5_smoke | qwen3.5-9b | 1 | 0 | 0.0% | 0.0% | robber_tile, node_occupancy |
| 20260514T225638Z_qwen3.5_smoke | qwen3.5-9b | 1 | 1 | 0.0% | 0.0% | robber_tile, node_occupancy |
| 20260514T225916Z_qwen3_vs_qwen3.5_smoke | qwen3-vl-8b | 4 | 4 | 0.0% | 0.0% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260514T225916Z_qwen3_vs_qwen3.5_smoke | qwen3.5-9b | 4 | 4 | 0.0% | 33.3% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260514T225927Z_qwen3.5_vision_sweep | qwen3.5-9b | 12 | 12 | 0.0% | 33.3% | robber_tile, tile_resource_number, node_occupancy, edge_road_owner, port_type_nodes, longest_road_holder |
| 20260514T230048Z_qwen3_vs_qwen3.5_full_suite | qwen3-vl-8b | 130 | 130 | 15.4% | 13.9% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| 20260514T230048Z_qwen3_vs_qwen3.5_full_suite | qwen3.5-9b | 130 | 130 | 14.6% | 21.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| 20260514T231251Z_qwen3_vs_qwen3.5_full_suite_nothink9b | qwen3-vl-8b | 130 | 130 | 15.4% | 15.5% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| 20260514T231251Z_qwen3_vs_qwen3.5_full_suite_nothink9b | qwen3.5-9b | 130 | 130 | 16.9% | 21.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| 20260514T231609Z_qwen3_vs_qwen3.5_full_suite_nothink9b2 | qwen3-vl-8b | 130 | 0 | 0.0% | 0.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| 20260514T231609Z_qwen3_vs_qwen3.5_full_suite_nothink9b2 | qwen3.5-9b | 130 | 0 | 0.0% | 0.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| 20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3 | qwen3-vl-8b | 130 | 130 | 16.2% | 15.5% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| 20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3 | qwen3.5-9b | 130 | 130 | 15.4% | 21.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| 20260514T234814Z_gemini3p_numbers | gemini-3.1-pro-preview | 60 | 60 | 10.0% | 14.0% | tile_resource_number, robber_resource_number, color_road_count, color_building_counts, robber_tile, robber_adjacent_buildings |
| 20260514T234814Z_gemini3p_numbers | gemini-pro-latest | 60 | 60 | 15.0% | 19.6% | tile_resource_number, robber_resource_number, color_road_count, color_building_counts, robber_tile, robber_adjacent_buildings |
| 20260514T235044Z_gemini3p_numbers_full | gemini-3.1-pro-preview | 60 | 60 | 13.3% | 16.1% | tile_resource_number, robber_resource_number, color_road_count, color_building_counts, robber_tile, robber_adjacent_buildings |
| 20260514T235448Z_gemini3p_combo5 | gemini-3.1-pro-preview | 30 | 30 | 16.7% | 23.1% | tile_resource_number, robber_resource_number, color_road_count, color_building_counts, robber_tile, robber_adjacent_buildings |
| 20260514T235448Z_gemini3p_combo5 | gemini-3.1-pro-preview-customtools | 30 | 30 | 6.7% | 12.3% | tile_resource_number, robber_resource_number, color_road_count, color_building_counts, robber_tile, robber_adjacent_buildings |
| 20260514T235645Z | gemini-3.1-pro-preview | 40 | 0 | 0.0% | 0.0% | tile_resource_number, robber_resource_number |
| 20260514T235645Z | gemini-3.1-pro-preview-customtools | 40 | 0 | 0.0% | 0.0% | tile_resource_number, robber_resource_number |
| 20260514T235657Z | gemini-3.1-pro-preview | 1 | 0 | 0.0% | 0.0% | tile_resource_number |
| 20260515T020142Z | qwen_qwen3.6-flash | 60 | 0 | 0.0% | 0.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| 20260515T022305Z_qwen3.6_flash_full_suite | qwen_qwen3.6-flash | 130 | 0 | 0.0% | 0.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| 20260515T022320Z_qwen3.6_flash_full_suite | qwen_qwen3.6-flash | 130 | 128 | 39.1% | 37.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| 20260515T024908Z_qwen3.6_flash_full_suite | qwen_qwen3.6-flash | 130 | 130 | 20.0% | 20.6% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| claude_fable_5_20260810 | anthropic_claude-fable-5 | 110 | 110 | 30.0% | 32.1% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, color_building_counts, color_road_count |
| claude_fable_5_budget1024_smoke_20260810 | anthropic_claude-fable-5 | 11 | 11 | 54.5% | 61.1% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, color_building_counts, color_road_count |
| claude_fable_5_smoke_20260810 | anthropic_claude-fable-5 | 1 | 1 | 100.0% | 100.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, color_building_counts, color_road_count |
| manual_moondream_native_smoke | moondream2 | 1 | 0 | 0.0% | 0.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| manual_moondream_native_smoke_fix | moondream2 | 1 | 0 | 0.0% | 0.0% | robber_tile |
| manual_moondream_native_smoke_fix2 | moondream2 | 1 | 0 | 0.0% | 0.0% | robber_tile |
| manual_moondream_smoke | moondream_moondream2 | 1 | 0 | 0.0% | 0.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| manual_qwen2_5_vl_2b_smoke | qwen_qwen2.5-vl-2b-instruct | 3 | 0 | 0.0% | 0.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| moondream2_free_smoke | openrouter_moondream2_free | 1 | 0 | 0.0% | 0.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| moondream2_smoke | openrouter_moondream2 | 5 | 0 | 0.0% | 0.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| moondream2_smoke2 | moondream_moondream2 | 3 | 0 | 0.0% | 0.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| qwen2_5_vl_2b_smoke | qwen_qwen2.5-vl-2b-instruct | 3 | 0 | 0.0% | 0.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| qwen3_8_27b_smoke_20260816 | qwen_qwen3.8-27b | 11 | 11 | 45.5% | 44.4% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, color_building_counts, color_road_count |
| qwen3_8_27b_visual_20260816 | qwen_qwen3.8-27b | 110 | 110 | 21.8% | 20.2% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, color_building_counts, color_road_count |
| qwen3_instruct_smoke2 | qwen_qwen3-vl-8b-instruct | 1 | 0 | 0.0% | 0.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |
| qwen3_vl_32b_dense_20260810 | qwen_qwen3-vl-32b-instruct | 110 | 110 | 15.5% | 14.7% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, color_building_counts, color_road_count |
| qwen3v8b_probe | qwen_qwen3-vl-8b | 1 | 0 | 0.0% | 0.0% | robber_tile, robber_resource_number, tile_resource_number, tile_has_robber, node_occupancy, edge_road_owner, color_road_locations, port_trade_type, port_occupancy, nodes_connected, edge_connects_nodes, color_building_counts, color_road_count |

## Qwen3-VL-8B Category Breakdown

| Run | Category | Attempted | Exact | Component | Errors |
| --- | --- | --- | --- | --- | --- |
| 20260513T200550Z | edge_road_owner | 2 | 0.0% | 0.0% | 0 |
| 20260513T200550Z | longest_road_holder | 2 | 100.0% | 100.0% | 0 |
| 20260513T200550Z | node_occupancy | 2 | 100.0% | 100.0% | 0 |
| 20260513T200550Z | port_type_nodes | 2 | 0.0% | 62.5% | 0 |
| 20260513T200550Z | robber_tile | 2 | 50.0% | 50.0% | 0 |
| 20260513T200550Z | tile_resource_number | 2 | 0.0% | 25.0% | 0 |
| 20260513T201207Z | edge_road_owner | 5 | 60.0% | 60.0% | 0 |
| 20260513T201207Z | longest_road_holder | 5 | 80.0% | 80.0% | 0 |
| 20260513T201207Z | node_occupancy | 5 | 80.0% | 66.7% | 0 |
| 20260513T201207Z | port_type_nodes | 5 | 0.0% | 65.0% | 0 |
| 20260513T201207Z | robber_tile | 5 | 40.0% | 40.0% | 0 |
| 20260513T201207Z | tile_resource_number | 5 | 0.0% | 0.0% | 0 |
| 20260513T212010Z | color_building_counts | 5 | 0.0% | 0.0% | 0 |
| 20260513T212010Z | edge_road_owner | 5 | 0.0% | 0.0% | 0 |
| 20260513T212010Z | node_occupancy | 5 | 0.0% | 0.0% | 0 |
| 20260513T212010Z | port_trade_type | 5 | 60.0% | 60.0% | 0 |
| 20260513T212010Z | robber_resource_number | 5 | 0.0% | 0.0% | 0 |
| 20260513T212010Z | robber_tile | 5 | 0.0% | 0.0% | 0 |
| 20260513T212010Z | tile_has_robber | 5 | 100.0% | 100.0% | 0 |
| 20260513T212010Z | tile_resource_number | 5 | 0.0% | 0.0% | 0 |
| 20260513T212255Z | color_building_counts | 5 | 0.0% | 0.0% | 0 |
| 20260513T212255Z | edge_road_owner | 5 | 0.0% | 0.0% | 0 |
| 20260513T212255Z | node_occupancy | 5 | 0.0% | 0.0% | 0 |
| 20260513T212255Z | port_trade_type | 5 | 60.0% | 60.0% | 0 |
| 20260513T212255Z | robber_resource_number | 5 | 0.0% | 0.0% | 0 |
| 20260513T212255Z | robber_tile | 5 | 0.0% | 0.0% | 0 |
| 20260513T212255Z | tile_has_robber | 5 | 0.0% | 0.0% | 0 |
| 20260513T212255Z | tile_resource_number | 5 | 0.0% | 0.0% | 0 |
| 20260514T222811Z_qwen3_vs_qwen3.5 | edge_road_owner | 0 | 0.0% | 0.0% | 15 |
| 20260514T222811Z_qwen3_vs_qwen3.5 | node_occupancy | 0 | 0.0% | 0.0% | 20 |
| 20260514T222811Z_qwen3_vs_qwen3.5 | port_type_nodes | 0 | 0.0% | 0.0% | 10 |
| 20260514T222811Z_qwen3_vs_qwen3.5 | robber_tile | 0 | 0.0% | 0.0% | 5 |
| 20260514T222811Z_qwen3_vs_qwen3.5 | tile_resource_number | 0 | 0.0% | 0.0% | 15 |
| 20260514T222854Z_qwen3_vs_qwen3.5 | edge_road_owner | 5 | 0.0% | 0.0% | 0 |
| 20260514T222854Z_qwen3_vs_qwen3.5 | node_occupancy | 5 | 0.0% | 0.0% | 0 |
| 20260514T222854Z_qwen3_vs_qwen3.5 | port_type_nodes | 5 | 0.0% | 65.0% | 0 |
| 20260514T222854Z_qwen3_vs_qwen3.5 | robber_tile | 5 | 20.0% | 20.0% | 0 |
| 20260514T222854Z_qwen3_vs_qwen3.5 | tile_resource_number | 10 | 0.0% | 5.0% | 0 |
| 20260514T225916Z_qwen3_vs_qwen3.5_smoke | robber_tile | 2 | 0.0% | 0.0% | 0 |
| 20260514T225916Z_qwen3_vs_qwen3.5_smoke | tile_resource_number | 2 | 0.0% | 0.0% | 0 |
| 20260514T230048Z_qwen3_vs_qwen3.5_full_suite | color_building_counts | 10 | 0.0% | 10.0% | 0 |
| 20260514T230048Z_qwen3_vs_qwen3.5_full_suite | color_road_count | 10 | 0.0% | 0.0% | 0 |
| 20260514T230048Z_qwen3_vs_qwen3.5_full_suite | color_road_locations | 10 | 0.0% | 10.4% | 0 |
| 20260514T230048Z_qwen3_vs_qwen3.5_full_suite | edge_connects_nodes | 10 | 50.0% | 50.0% | 0 |
| 20260514T230048Z_qwen3_vs_qwen3.5_full_suite | edge_road_owner | 10 | 0.0% | 0.0% | 0 |
| 20260514T230048Z_qwen3_vs_qwen3.5_full_suite | node_occupancy | 10 | 0.0% | 0.0% | 0 |
| 20260514T230048Z_qwen3_vs_qwen3.5_full_suite | nodes_connected | 10 | 50.0% | 50.0% | 0 |
| 20260514T230048Z_qwen3_vs_qwen3.5_full_suite | port_occupancy | 10 | 50.0% | 25.0% | 0 |
| 20260514T230048Z_qwen3_vs_qwen3.5_full_suite | port_trade_type | 10 | 50.0% | 50.0% | 0 |
| 20260514T230048Z_qwen3_vs_qwen3.5_full_suite | robber_resource_number | 10 | 0.0% | 3.3% | 0 |
| 20260514T230048Z_qwen3_vs_qwen3.5_full_suite | robber_tile | 10 | 0.0% | 0.0% | 0 |
| 20260514T230048Z_qwen3_vs_qwen3.5_full_suite | tile_has_robber | 10 | 0.0% | 0.0% | 0 |
| 20260514T230048Z_qwen3_vs_qwen3.5_full_suite | tile_resource_number | 10 | 0.0% | 0.0% | 0 |
| 20260514T231251Z_qwen3_vs_qwen3.5_full_suite_nothink9b | color_building_counts | 10 | 0.0% | 20.0% | 0 |
| 20260514T231251Z_qwen3_vs_qwen3.5_full_suite_nothink9b | color_road_count | 10 | 0.0% | 0.0% | 0 |
| 20260514T231251Z_qwen3_vs_qwen3.5_full_suite_nothink9b | color_road_locations | 10 | 0.0% | 10.4% | 0 |
| 20260514T231251Z_qwen3_vs_qwen3.5_full_suite_nothink9b | edge_connects_nodes | 10 | 50.0% | 50.0% | 0 |
| 20260514T231251Z_qwen3_vs_qwen3.5_full_suite_nothink9b | edge_road_owner | 10 | 0.0% | 0.0% | 0 |
| 20260514T231251Z_qwen3_vs_qwen3.5_full_suite_nothink9b | node_occupancy | 10 | 0.0% | 0.0% | 0 |
| 20260514T231251Z_qwen3_vs_qwen3.5_full_suite_nothink9b | nodes_connected | 10 | 50.0% | 50.0% | 0 |
| 20260514T231251Z_qwen3_vs_qwen3.5_full_suite_nothink9b | port_occupancy | 10 | 50.0% | 25.0% | 0 |
| 20260514T231251Z_qwen3_vs_qwen3.5_full_suite_nothink9b | port_trade_type | 10 | 50.0% | 50.0% | 0 |
| 20260514T231251Z_qwen3_vs_qwen3.5_full_suite_nothink9b | robber_resource_number | 10 | 0.0% | 10.0% | 0 |
| 20260514T231251Z_qwen3_vs_qwen3.5_full_suite_nothink9b | robber_tile | 10 | 0.0% | 0.0% | 0 |
| 20260514T231251Z_qwen3_vs_qwen3.5_full_suite_nothink9b | tile_has_robber | 10 | 0.0% | 0.0% | 0 |
| 20260514T231251Z_qwen3_vs_qwen3.5_full_suite_nothink9b | tile_resource_number | 10 | 0.0% | 0.0% | 0 |
| 20260514T231609Z_qwen3_vs_qwen3.5_full_suite_nothink9b2 | color_building_counts | 0 | 0.0% | 0.0% | 10 |
| 20260514T231609Z_qwen3_vs_qwen3.5_full_suite_nothink9b2 | color_road_count | 0 | 0.0% | 0.0% | 10 |
| 20260514T231609Z_qwen3_vs_qwen3.5_full_suite_nothink9b2 | color_road_locations | 0 | 0.0% | 0.0% | 10 |
| 20260514T231609Z_qwen3_vs_qwen3.5_full_suite_nothink9b2 | edge_connects_nodes | 0 | 0.0% | 0.0% | 10 |
| 20260514T231609Z_qwen3_vs_qwen3.5_full_suite_nothink9b2 | edge_road_owner | 0 | 0.0% | 0.0% | 10 |
| 20260514T231609Z_qwen3_vs_qwen3.5_full_suite_nothink9b2 | node_occupancy | 0 | 0.0% | 0.0% | 10 |
| 20260514T231609Z_qwen3_vs_qwen3.5_full_suite_nothink9b2 | nodes_connected | 0 | 0.0% | 0.0% | 10 |
| 20260514T231609Z_qwen3_vs_qwen3.5_full_suite_nothink9b2 | port_occupancy | 0 | 0.0% | 0.0% | 10 |
| 20260514T231609Z_qwen3_vs_qwen3.5_full_suite_nothink9b2 | port_trade_type | 0 | 0.0% | 0.0% | 10 |
| 20260514T231609Z_qwen3_vs_qwen3.5_full_suite_nothink9b2 | robber_resource_number | 0 | 0.0% | 0.0% | 10 |
| 20260514T231609Z_qwen3_vs_qwen3.5_full_suite_nothink9b2 | robber_tile | 0 | 0.0% | 0.0% | 10 |
| 20260514T231609Z_qwen3_vs_qwen3.5_full_suite_nothink9b2 | tile_has_robber | 0 | 0.0% | 0.0% | 10 |
| 20260514T231609Z_qwen3_vs_qwen3.5_full_suite_nothink9b2 | tile_resource_number | 0 | 0.0% | 0.0% | 10 |
| 20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3 | color_building_counts | 10 | 0.0% | 5.0% | 0 |
| 20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3 | color_road_count | 10 | 10.0% | 10.0% | 0 |
| 20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3 | color_road_locations | 10 | 0.0% | 12.5% | 0 |
| 20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3 | edge_connects_nodes | 10 | 50.0% | 50.0% | 0 |
| 20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3 | edge_road_owner | 10 | 0.0% | 0.0% | 0 |
| 20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3 | node_occupancy | 10 | 0.0% | 0.0% | 0 |
| 20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3 | nodes_connected | 10 | 50.0% | 50.0% | 0 |
| 20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3 | port_occupancy | 10 | 50.0% | 25.0% | 0 |
| 20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3 | port_trade_type | 10 | 50.0% | 50.0% | 0 |
| 20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3 | robber_resource_number | 10 | 0.0% | 10.0% | 0 |
| 20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3 | robber_tile | 10 | 0.0% | 0.0% | 0 |
| 20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3 | tile_has_robber | 10 | 0.0% | 0.0% | 0 |
| 20260514T231751Z_qwen3_vs_qwen3.5_full_suite_nothink9b3 | tile_resource_number | 10 | 0.0% | 5.0% | 0 |

## Qwen3-VL-8B Failure Buckets

| Category | Misses | Common expected | Common response | Example question IDs |
| --- | --- | --- | --- | --- |
| color_building_counts | 50 | <BLUE> SETTLEMENTS 1 CITIES 0 | 2 | sample_000_q21_color_building_counts; sample_001_q21_color_building_counts; sample_002_q21_color_building_counts |
| color_road_count | 39 | <BLUE> ROADS 1 | 2 | sample_000_q24_color_road_count; sample_001_q24_color_road_count; sample_002_q24_color_road_count |
| color_road_locations | 40 | <E19_46> |  | sample_000_q25_color_road_locations; sample_001_q25_color_road_locations; sample_002_q25_color_road_locations |
| edge_connects_nodes | 25 | YES | NO | sample_000_q29_edge_connects_nodes; sample_002_q29_edge_connects_nodes; sample_004_q29_edge_connects_nodes |
| edge_road_owner | 74 | <BLUE> | EMPTY | sample_000_q10_edge_road_owner; sample_001_q10_edge_road_owner; sample_000_q10_edge_road_owner |
| longest_road_holder | 1 | NONE | <T07> | sample_004_q01_longest_road_holder |
| node_occupancy | 76 | <RED> <SETTLEMENT> | EMPTY | sample_004_q07_node_occupancy; sample_000_q09_node_occupancy; sample_001_q09_node_occupancy |
| nodes_connected | 25 | YES | NO | sample_000_q28_nodes_connected; sample_002_q28_nodes_connected; sample_004_q28_nodes_connected |
| port_occupancy | 25 | NONE | NONE | sample_001_q19_port_occupancy; sample_003_q19_port_occupancy; sample_004_q19_port_occupancy |
| port_trade_type | 29 | <SHEEP> 2:1 | GENERIC 3:1 | sample_000_q17_port_trade_type; sample_002_q17_port_trade_type; sample_000_q17_port_trade_type |
| port_type_nodes | 22 | GENERIC 3:1 <N52> <N53> |  | sample_000_q13_port_type_nodes; sample_001_q13_port_type_nodes; sample_000_q13_port_type_nodes |
| robber_resource_number | 50 | <T17> <BRICK> 9 | <T03> <WOOD> 3 | sample_000_q01_robber_resource_number; sample_001_q01_robber_resource_number; sample_002_q01_robber_resource_number |
| robber_tile | 65 | <T17> | <T03> | sample_000_q00_robber_tile; sample_001_q00_robber_tile; sample_003_q00_robber_tile |
| tile_has_robber | 45 | YES | NO | sample_000_q07_tile_has_robber; sample_001_q07_tile_has_robber; sample_002_q07_tile_has_robber |
| tile_resource_number | 84 | <WOOD> 4 | <DESERT> NO_NUMBER | sample_000_q04_tile_resource_number; sample_001_q04_tile_resource_number; sample_000_q04_tile_resource_number |

Interpretation target for SFT: exact accuracy should rise first on atlas-bound local grounding tasks (`tile_resource_number`, `robber_tile`, `edge_road_owner`, `port_type_nodes`) without only improving sentinel-heavy categories such as `longest_road_holder` or empty-node answers.
