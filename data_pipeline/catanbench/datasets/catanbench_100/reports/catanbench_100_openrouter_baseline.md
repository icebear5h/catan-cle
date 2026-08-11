# CatanBench-100 OpenRouter Baseline

Scope: stored OpenRouter eval runs under `data_pipeline/catanbench/datasets/catanbench_100/openrouter_eval`. These are the frozen baseline numbers before any Catan-specific tuning.

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

## Qwen3-VL-8B Failure Buckets

| Category | Misses | Common expected | Common response | Example question IDs |
| --- | --- | --- | --- | --- |
| color_building_counts | 10 | <BLUE> SETTLEMENTS 1 CITIES 0 | 2 | sample_000_q21_color_building_counts; sample_001_q21_color_building_counts; sample_002_q21_color_building_counts |
| edge_road_owner | 14 | <BLUE> | EMPTY | sample_000_q10_edge_road_owner; sample_001_q10_edge_road_owner; sample_000_q10_edge_road_owner |
| longest_road_holder | 1 | NONE | <T07> | sample_004_q01_longest_road_holder |
| node_occupancy | 11 | <RED> <SETTLEMENT> | EMPTY | sample_004_q07_node_occupancy; sample_000_q09_node_occupancy; sample_001_q09_node_occupancy |
| port_trade_type | 4 | <SHEEP> 2:1 | GENERIC 3:1 | sample_000_q17_port_trade_type; sample_002_q17_port_trade_type; sample_000_q17_port_trade_type |
| port_type_nodes | 7 | <SHEEP> 2:1 <N47> <N45> | GENERIC;<N47> <N45> | sample_000_q13_port_type_nodes; sample_001_q13_port_type_nodes; sample_000_q13_port_type_nodes |
| robber_resource_number | 10 | <T17> <BRICK> 9 | <T03> <WOOD> 3 | sample_000_q01_robber_resource_number; sample_001_q01_robber_resource_number; sample_002_q01_robber_resource_number |
| robber_tile | 14 | <T17> | <T03> | sample_000_q00_robber_tile; sample_001_q00_robber_tile; sample_003_q00_robber_tile |
| tile_has_robber | 5 | YES | NO | sample_000_q07_tile_has_robber; sample_001_q07_tile_has_robber; sample_002_q07_tile_has_robber |
| tile_resource_number | 17 | <WOOD> 11 | <DESERT> NO_NUMBER | sample_000_q04_tile_resource_number; sample_001_q04_tile_resource_number; sample_000_q04_tile_resource_number |

Interpretation target for SFT: exact accuracy should rise first on atlas-bound local grounding tasks (`tile_resource_number`, `robber_tile`, `edge_road_owner`, `port_type_nodes`) without only improving sentinel-heavy categories such as `longest_road_holder` or empty-node answers.
