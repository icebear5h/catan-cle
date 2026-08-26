# CatanBoardBench-100 Initial Findings

Date: 2026-05-13

## Summary

Initial off-the-shelf small VLM results show broad failure on local Catan board
grounding. The models can sometimes answer easy sentinel/global questions, but
they do not reliably bind visible board facts to tile, edge, node, and port IDs.

| Category | Exact acc | Component acc | Read |
| --- | ---: | ---: | --- |
| tile_resource_number | 8.3% | 22.9% | Broad failure. Models cannot reliably bind tile ID to visible resource/number. |
| robber_tile | 16.7% | 16.7% | Broad failure. Often off by nearby/wrong tile. |
| edge_road_owner | 17.4% | 17.4% | Broad failure. Occupied roads often called EMPTY or wrong color. |
| port_type_nodes | 0.0% | 57.6% | Partial success. Models often get node IDs, but miss ratio/resource type. |
| node_occupancy | 73.9% | 66.7% | Easier, but many queried nodes were empty. Needs more positive examples. |
| longest_road_holder | 95.7% | 95.7% | Probably too easy because most answers were NONE. |

## Difficulty Hierarchy

1. Easy: global/simple sentinel answers like `NONE`, and many empty-node queries.
2. Medium: node occupancy, especially when the queried node is empty.
3. Hard: ports, because they require icon/type recognition plus atlas node binding.
4. Very hard: tile ID to resource/number, robber tile, and edge road owner.

## Interpretation

The main conclusion is not that Catan strategy is hard yet. These failures happen
before policy reasoning. Raw small VLMs are weak at local board grounding under
the current board-image benchmark.

This supports the project thesis:

- "Describe the board" is too vague for Catan.
- Evaluation needs explicit node/edge/tile/port grounding.
- Board perception must be separated from policy/action quality.
- SFT should teach the atlas and visual bindings directly before strategy SFT or
  RL.

## Benchmark Follow-Ups

The current aggregate scores should be treated as a first diagnostic, not a final
benchmark because some categories are class-imbalanced.

Recommended fixes:

- Balance node occupancy across empty, settlement, and city examples.
- Balance longest-road-holder examples across `NONE` and each player color.
- Report separate positive/negative accuracy for road ownership and node
  occupancy.
- Keep exact-match and component-match scores separate.
- Add condition tags such as `colonist_512`, `colonist_768`, `clean_512`,
  `clean_with_ids`, and `local_crop`.

## Training Implications

For off-the-shelf model benchmarks, continue using plain labels such as `node
42`, `edge 12-17`, `tile 8`, and `port 3`.

For open-weight training, use custom atlas tokens after tokenizer extension:

- `<N00>` through `<N53>`
- canonical edge tokens like `<E12_17>`
- `<T00>` through `<T18>`
- `<P00>` through `<P08>`
- `<WOOD>`, `<BRICK>`, `<SHEEP>`, `<WHEAT>`, `<ORE>`, `<DESERT>`
- `<ROBBER>`

Suggested training ladder:

1. Atlas-only text QA to teach topology and IDs.
2. Clean synthetic render QA with explicit labels.
3. Clean render QA without labels.
4. Colonist-style board QA at 768px and 512px.
5. Local crops for roads, ports, tile numbers, and robber position.
6. Policy SFT only after board-contract extraction is reliable.
