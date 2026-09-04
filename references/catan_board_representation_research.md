# Catan Board Representation Research

Date: 2026-08-11

## Executive decision

Use one authoritative, typed **board cell-complex contract**, then render it through
model-specific adapters. Do not make images, prose, JSON, or a neural tensor the
source of truth.

For the base game, split board information into three lifetimes:

1. **Atlas-static:** 19 land slots, 54 corner slots, 72 road slots, nine port
   slots, coordinates, and all incidence relations.
2. **Episode-static:** which resource and number occupy each land slot, and
   which trade type occupies each port slot.
3. **Decision-dynamic:** buildings, roads, robber, player/public state,
   perspective-private state, and legal actions.

This resolves the port issue directly: a port **location** is stable in the base
atlas, while its **trade type** is an episode attribute. A port should be a typed
entity attached to one coastal road and its two endpoint corners, not a property
memorized as `P03 = wheat`.

Recommended feeds:

- **Frontier/black-box teacher:** compact sectioned Catan DSL, full resource
  words, canonical IDs, self-describing legal candidates, and an indexed action
  menu. An image may be auxiliary, never authoritative.
- **Trainable language-model student:** the same DSL, but with trained atomic
  atlas tokens such as `<T08>`, `<N19>`, and `<E19_46>`.
- **Neural RL baseline or future board adapter:** a heterogeneous incidence
  graph/entity set derived from the same contract, with a scorer over legal
  action candidates.

The key design is therefore not “pick the best coordinate.” Use three layers at
once:

- canonical ID for exact reference and action execution;
- static coordinate/incidence data for geometry;
- dynamic Catan-language descriptor for model intuition.

## What is fixed and what varies in this repository

The base topology in `cle/game_engine/models/map.py` contains:

- 19 land tile slots;
- 54 playable corner/node slots;
- 72 playable road/edge slots;
- nine fixed port slots on the outer ring;
- one 2:1 port for each resource and four generic 3:1 ports;
- four wood, three brick, four sheep, four wheat, three ore, and one desert;
- one 2 and one 12, plus two of each number 3–6 and 8–11.

`initialize_tiles()` independently shuffles the resource list, number list, and
port-type list, then assigns IDs while iterating the ordered topology. Therefore:

- `T00`, `N00`, and `P00` are de facto stable only while topology insertion
  order and ID-generation logic stay unchanged;
- a tile slot's coordinate and neighbors are fixed, but its resource, number,
  and robber status are not;
- a port slot's coordinate, coastal edge, and two corner endpoints are fixed,
  but its 2:1 resource or 3:1 generic type is not;
- buildings, roads, and robber position change during play.

The exact base port geometry currently is:

```text
P00 cube( 3,-3, 0) coast E25_26 corners N25,N26
P01 cube( 1,-3, 2) coast E28_29 corners N28,N29
P02 cube(-1,-2, 3) coast E32_33 corners N32,N33
P03 cube(-3, 0, 3) coast E35_36 corners N35,N36
P04 cube(-3, 2, 1) coast E38_39 corners N38,N39
P05 cube(-2, 3,-1) coast E40_44 corners N40,N44
P06 cube( 0, 3,-3) coast E45_47 corners N47,N45
P07 cube( 2, 1,-3) coast E48_49 corners N48,N49
P08 cube( 3,-1,-2) coast E52_53 corners N52,N53
```

Every attached corner pair is also a playable coastal road edge. Strategically,
that coastal edge and its endpoints are a more useful port location than the
water-hex coordinate alone.

### Important generation caveat

The repository's base generator samples and pops the number list. It does not
visibly enforce an official-style “no adjacent 6 and 8” constraint. A board
contract must describe the board actually produced by the engine, not assume a
rule that is absent from generation.

## Existing representation patterns in the codebase

### 1. Authoritative engine object graph

Relevant files:

- `cle/game_engine/models/map.py`
- `cle/game_engine/models/board.py`
- `cle/game_engine/models/coordinate_system.py`

The engine already has the right semantic decomposition: land faces, graph
vertices, graph edges, ports, cube coordinates, and cached incidence lookups.
This is the correct oracle.

Strengths:

- exact rules-compatible state;
- direct adjacency and reachability;
- compact in memory;
- separates land tiles, ports, nodes, and edges.

Hazards:

- IDs are an emergent consequence of ordered dictionary iteration, not an
  explicit versioned ABI;
- edges appear in both orientations in some engine caches;
- the static graph is created from the base topology and is not a general
  per-atlas graph boundary;
- `None` means desert for land tiles but generic 3:1 for ports.

### 2. Flat/tensor ML projections

Relevant files:

- `cle/game_engine/features.py`
- upstream Catanatron `features.py` and `gym/board_tensor_features.py`

The current feature vector creates fixed slots such as `TILE3_IS_WOOD`,
`PORT2_IS_ORE`, `NODE9_P2_CITY`, and `EDGE(9,10)_P1_ROAD`. For four players,
the upstream Catanatron raw observation formula is `194*N + 226`, or 1,002
features.

Catanatron's newer mixed representation uses a board tensor with shape
`(2*N + 12, 21, 11)` plus non-spatial numeric features. At four players, that
is 20 board channels. It contains:

- one building and one road channel per relative player;
- five resource-production channels;
- one robber channel;
- six port channels.

This is an important baseline. It also illustrates a tradeoff: its resource
planes store roll probability at corners rather than exact dice-number identity,
so 6 and 8 are deliberately collapsed. That can be useful for policy learning,
but it is not a lossless language/replay contract.

Strengths:

- fast and easy to batch;
- fixed shapes work with standard MLP/CNN code;
- relative-player channels support one shared policy;
- mature legal-action masking exists alongside it.

Weaknesses:

- the flat form has weak relational inductive bias;
- the rectangular “brick” embedding is sparse and topology-specific;
- port and tile fields can be lossy if reduced to derived probabilities;
- map variants require shape/schema decisions;
- neither representation is intuitive input for an off-the-shelf LLM.

### 3. Explicit CatanBoardBench contracts and atlas tokens

Relevant files:

- `evals/catan_board_bench/tokens.py`
- `evals/catan_board_bench/builder.py`
- `evals/catan_board_bench/annotations.py`

CatanBoardBench already provides the best starting boundary:

- `<T00>`–`<T18>` for tiles;
- `<N00>`–`<N53>` for corners;
- endpoint-based edge tokens such as `<E19_46>`;
- `<P00>`–`<P08>` for port slots;
- a public board contract with tiles, nodes, edges, ports, robber, players, and
  awards.

Strengths:

- typed namespaces;
- engine-derived topology;
- canonical edge endpoints;
- exact and auditable;
- usable by rendering, QA, and training pipelines.

Weaknesses:

- `catan_public_board_contract/v0` is intentionally verbose and highly
  redundant;
- it has a contract version but no separately versioned/hashed atlas ABI;
- it is suitable as an oracle/debug artifact, not as-is as the model prompt.

### 4. Current semantic LLM formatter

Relevant files:

- `cle/env/observation_formatter.py`
- `cle/harness/replay.py`

The formatter gives readable descriptions of owned and opposing placements and
rich action descriptions. The separate replay decision-packet builder enumerates
every engine `playable_action` into one authoritative indexed menu. By contrast,
the formatter's embedded `<valid_actions>` section is only a summary: it drops
string-valued meta-actions and truncates most non-setup groups to five examples.
It must never be used as the executable action set.

Current limitations for board reasoning:

- the observation does not enumerate the full tile and port setup;
- own/opponent roads are summarized by count rather than exact locations;
- setup understanding is carried mostly by a long repeated legal-action menu;
- node coordinates are derived by summing incident tile coordinates, but this
  coordinate system is not named or versioned;
- `_check_nearby_opponents()` treats numeric ID distance as approximate spatial
  distance, which is not topologically valid;
- action descriptions can repeat large natural-language neighborhoods while
  raw engine dataclass strings carry the actual target ID.

The summed vertex coordinate is still useful: an empirical check found 54
unique triples for the 54 playable base corners. It should be named something
like `vertex_cube3/v1`, documented as a base-atlas coordinate, and generated
from the atlas—not presented as though it were the same coordinate type used
for tile centers.

## Repository correctness hazards to resolve before freezing a schema

These are not reasons to discard the current atlas. They are reasons to make it
the single authority.

- A removed `cle/env/node_positions.py` prototype said playable nodes were
  0–53 but classified IDs through 95. It was deleted because its “coastal” set
  did not describe the playable-node namespace.
- The removed legacy `data_pipeline/bootstrapping/colonist/board_layout.py`
  claimed 54 nodes, but its manual tile mapping reached only node 43 and its
  hard-coded ports used that incompatible namespace.
- Engine IDs, Colonist tile/corner/edge IDs, cube coordinates, and endpoint-edge
  IDs are separate namespaces. Replay prose currently exposes unqualified
  Colonist coordinates in some activity rows while model actions use engine
  objects.
- Colonist port parsing initializes unresolved positions to `None`; because
  `None` also means a valid generic port, an unresolved/unknown port can silently
  become 3:1.
- CatanBoardBench's playable-edge helper falls back to base edges if a map does not
  have 72 edges. A canonical contract should reject a mismatched atlas instead
  of silently projecting base topology.

Required rule: every external source reference must carry its namespace, and an
unknown port must stay `UNKNOWN`, never become `GENERIC`.

## Literature and environment evidence

### Catan-specific neural encodings

**Gendre and Kaneko, “Playing Catan with Cross-dimensional Neural Network”
(ICONIP 2020).**

Source: <https://arxiv.org/abs/2008.07079>

- Explicitly identifies Catan as a mixture of 19 faces, 72 paths, 54
  intersections, player cards, imperfect information, stochasticity, and a
  large heterogeneous action space.
- Embeds the board into a brick coordinate layout, keeps different object types
  in separate channels, and connects spatial processing to scalar features via
  cross-dimensional layers.
- Uses legal-action masking.
- The reported setup is two-player and does not learn player trading, so it is
  evidence for separating spatial and scalar channels—not proof of the best
  full four-player representation.

**Driss and Cazenave, “Deep Catan” (2022).**

Source: <https://www.lamsade.dauphine.fr/~cazenave/papers/DeepCatan.pdf>

- Uses a 23x13 brick-coordinate image with 29 board channels and a separate
  scalar vector.
- Board channels separate roads, settlements, cities, ports, resource types,
  resource odds, and robber odds.
- Also omits player-to-player trading. Its experiments do not establish that a
  CNN grid is superior to an entity/graph encoding.

**Charlesworth, “Learning to Play Settlers of Catan with Deep RL” (2021).**

Sources:

- <https://settlers-rl.github.io/>
- <https://github.com/henrycharlesworth/settlers_of_catan_RL>

This practitioner implementation is especially relevant:

- each of 19 tiles gets a 61-dimensional feature vector containing its number,
  resource, robber state, and the six adjacent-corner building/owner features;
- self-attention processes the tile entity set;
- players are represented relative to the actor (`self`, clockwise +1, +2,
  +3);
- opponent resources are represented as public-information minimum/maximum
  ranges rather than omniscient hands;
- structured conditional action heads cover type, corner, edge, tile, player,
  resource, and recurrent trade arguments, each with legality masks.

The author explicitly notes limited ablation capacity. Treat it as a strong
architecture precedent, not a controlled representation comparison.

**Catanatron.**

Sources:

- <https://docs.catanatron.com/advanced/data-and-machine-learning>
- <https://github.com/bcollazo/catanatron>

Catanatron supplies two valuable baselines:

- an exact fixed feature vector keyed by tile/node/edge/port IDs;
- a mixed 21x11 board tensor plus numeric features.

Its Gymnasium environment exposes a sorted reproducible discrete action mapping,
`valid_actions`, and a boolean `action_masks()` method. This supports keeping
state representation and legality as separate authoritative outputs.

### General relational and entity-centric evidence

**Battaglia et al., “Relational inductive biases, deep learning, and graph
networks” (2018).**

Source: <https://arxiv.org/abs/1806.01261>

The graph-network framework represents entities as nodes, relations as edges,
and system-level properties as global attributes. Per-node and per-edge
functions are reused, and set aggregations are permutation invariant. That is a
natural match for Catan's faces, vertices, edge slots, ports, players, and global
phase.

**Zambaldi et al., “Relational Deep Reinforcement Learning” (ICLR 2019).**

Source: <https://arxiv.org/abs/1806.01830>

Self-attention over entity representations improved sample efficiency and
zero-shot generalization on relational RL tasks. This supports entity tokens or
a graph transformer, but it does not decide Catan's entity schema for us.

**AlphaStar (Vinyals et al., Nature 2019).**

Sources:

- <https://www.nature.com/articles/s41586-019-1724-z>
- <https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf>

AlphaStar processed visible entities with self-attention, non-spatial data with
separate encoders, partial observability with memory, and a highly structured
action space using an autoregressive policy and pointer network. The analogy is
not that Catan needs AlphaStar's scale. It is that object/entity encoders plus
conditional target selection are a proven response to heterogeneous state and
action spaces.

**Graph-based Hex work.**

Source: <https://arxiv.org/abs/2311.13414>

GraphDQN experiments on Hex report better long-range dependency handling and
board-size transfer in some conditions, while strong fully convolutional models
remain competitive or stronger at local patterns. The useful conclusion is not
“GNN always wins”; it is that explicit topology is a valuable ablation against
image/grid representations.

### Environment API evidence

**OpenSpiel** separates player observations/information states from the list of
legal integer actions:

- <https://openspiel.readthedocs.io/en/latest/api_reference/state_observation_tensor.html>
- <https://openspiel.readthedocs.io/en/latest/api_reference/state_information_state_tensor.html>
- <https://openspiel.readthedocs.io/en/latest/api_reference/state_legal_actions.html>

**PettingZoo** recommends an observation dictionary containing the observation
and an `action_mask`; its action-masking tutorial describes masking as more
natural than letting illegal actions have no effect:

- <https://pettingzoo.farama.org/tutorials/custom_environment/3-action-masking/>

The direct implication is that the model should never manufacture raw geometry
or discover legality by failed execution. The harness supplies a perspective-safe
state and a complete legal candidate set.

### LLM serialization evidence

**Karbevska et al., “Lost in Serialization: Invariance and Generalization of LLM
Graph Reasoners” (2025 preprint).**

Source: <https://arxiv.org/abs/2511.10234>

- LLM graph reasoning is not inherently invariant to node relabeling, edge
  ordering, data structure, or syntax.
- Sorted/localized edge or adjacency representations are generally easier than
  randomly shuffled edge lists.
- Adjacency matrices can inflate sequence length to quadratic size.
- Fine-tuning can specialize a model to one serialization and make it brittle
  to another.
- JSON was preferred in some model/task combinations, but no surface syntax was
  universally best.

**He et al., “Does Prompt Formatting Have Any Impact on LLM Performance?”
(2024).**

Source: <https://arxiv.org/abs/2411.10541>

Across plain text, Markdown, YAML, and JSON, format rankings changed by model and
task; larger models were more robust. This is direct evidence against choosing
JSON by intuition alone.

**Set-of-Mark prompting.**

Source: <https://arxiv.org/abs/2310.11441>

Speakable marks can improve VLM grounding. For Catan, overlays or atlas labels
are useful curriculum/diagnostic tools. They do not make a screenshot an exact
state oracle.

## Representation options and tradeoffs

### Full public-board JSON

Use for:

- storage;
- replay/debugging;
- validation;
- dataset provenance;
- API round trips.

Do not use the current verbose CatanBoardBench contract directly as the default
prompt. It repeats IDs, tokens, adjacency, endpoints, and empty slots.

### Compact canonical JSON

Use when a tool/API requires a standard typed payload. It is much smaller after
separating the atlas from state. It remains punctuation-heavy and is not proven
to be the most accurate LLM input.

### Sectioned Catan DSL

Best initial text hypothesis. It can be exact, sparse, canonical, and readable:

```text
CATAN-STATE v1 atlas=base-v1 orientation=engine-v1
TURN self=BLUE actor=BLUE phase=SETUP prompt=BUILD_INITIAL_SETTLEMENT

TILES  # fixed T order; id=number-resource
T00=11-WOOD T01=6-SHEEP T02=3-WHEAT ... T08=DESERT ... T18=12-WOOD

PORTS  # fixed P slot; current episode type and attachment
P00=2:1-BRICK@E25_26(N25,N26)
P01=2:1-WHEAT@E28_29(N28,N29)
...
P08=3:1-GENERIC@E52_53(N52,N53)

PIECES  # unlisted board slots are empty
N11=BLACK:SETTLEMENT N13=RED:SETTLEMENT
N19=BLUE:SETTLEMENT N22=WHITE:SETTLEMENT
E11_32=BLACK:ROAD E13_34=RED:ROAD
E19_46=BLUE:ROAD E22_23=WHITE:ROAD
ROBBER=T08
```

For a trained tokenizer, the same form becomes:

```text
<T00>=11:<WOOD> ... <P00>=2:1:<BRICK>@<E25_26>
<N19>=<BLUE>:<SETTLEMENT> <E19_46>=<BLUE>:<ROAD>
<ROBBER>=<T08>
```

The current added-token helper maps a resource value of `None` to `<DESERT>`.
That helper is valid for land tiles only. A frozen schema needs an explicit
`<GENERIC_PORT>` token (and, for staging data, `<UNKNOWN_PORT>`) so generic,
desert, absent, and unresolved states can never collide.

For an untuned external model, keep plain labels and full words. Unknown custom
tokens only fragment and add unfamiliar embeddings.

### Natural prose

Useful for strategic summaries and event history, but not as the sole board
contract. Prose invites omissions, paraphrase variance, and hard-to-detect
contradictions. Generate prose from typed state, never parse prose back into
truth.

### ASCII board

Potentially intuitive to a person, but Catan's faces, vertices, roads, and ports
are difficult to align without wide whitespace. Tokenizers charge for layout,
and small spacing changes can alter perceived adjacency. Keep ASCII as a
visualization ablation, not the canonical feed.

### Rectangular tensor/CNN

A strong numeric baseline supported by Catan-specific work. It is efficient,
but it introduces a brick/offset embedding and is less natural for an LLM. It
should remain in the bake-off.

### Heterogeneous graph/entity set

The strongest structural hypothesis for a learned policy or board-prefix
encoder. It preserves object type and incidence, supports variable map variants,
and produces target-specific embeddings for legal actions. It costs more
engineering than a text DSL and should not block the first controlled format
experiment.

### Board image

Use only when perception is itself the task, or as optional redundant context.
The engine already knows the exact state, so converting it to pixels and asking
a VLM to reconstruct it adds a lossy bottleneck.

Internal CatanBoardBench evidence for **image plus an enabled atlas prompt** reinforces
this under the tested operational settings:

- `qwen/qwen3-vl-32b-instruct`: 17/110 exact (15.45%);
- `anthropic/claude-fable-5`: 33/110 exact (30.0%), with 61/110 responses hitting
  the 256-token completion cap.

Reports:

- `evals/catan_board_bench/datasets/catan_board_bench_100/reports/2026-08-10-qwen3-vl-32b-visual.md`
- `evals/catan_board_bench/datasets/catan_board_bench_100/reports/2026-08-10-claude-fable-5-visual.md`

These are operational results under the tested prompt/budget, not model ceilings.
They are still far below the reliability required for an authoritative state
feed.

## Recommended canonical contract

### `BoardAtlas/v1` — immutable topology

```json
{
  "schema": "catan-board-atlas/v1",
  "atlas_id": "base-4p/engine-v1",
  "topology_sha256": "...",
  "orientation": "engine-v1",
  "tiles": [
    {
      "id": "T00",
      "cube": [0, 0, 0],
      "corners_clockwise": ["N00", "N01", "N02", "N03", "N04", "N05"],
      "road_slots_clockwise": ["E00_01", "E01_02", "E02_03", "E03_04", "E04_05", "E00_05"]
    }
  ],
  "corners": [
    {
      "id": "N00",
      "vertex_cube3": [1, 1, -2],
      "adjacent_tiles": ["T00", "T05", "T06"],
      "adjacent_roads": ["E00_01", "E00_05", "E00_20"]
    }
  ],
  "roads": [
    {
      "id": "E25_26",
      "corners": ["N25", "N26"],
      "adjacent_tiles": ["T07"],
      "coastal": true
    }
  ],
  "port_slots": [
    {
      "id": "P00",
      "water_cube": [3, -3, 0],
      "coast_road": "E25_26",
      "corners": ["N25", "N26"]
    }
  ]
}
```

The exact corner ordering should be generated and frozen from the engine. The
example above illustrates shape; implementation must use the generated atlas,
not hand-copied values.

### `BoardSetup/v1` — immutable within one game

```json
{
  "schema": "catan-board-setup/v1",
  "atlas_id": "base-4p/engine-v1",
  "atlas_sha256": "...",
  "setup_sha256": "...",
  "tiles": [
    {"id": "T00", "resource": "WOOD", "number": 11},
    {"id": "T08", "resource": "DESERT", "number": null}
  ],
  "ports": [
    {
      "id": "P00",
      "status": "PRESENT",
      "kind": "RESOURCE",
      "ratio": 2,
      "resource": "BRICK"
    },
    {
      "id": "P03",
      "status": "PRESENT",
      "kind": "GENERIC",
      "ratio": 3,
      "resource": null
    }
  ]
}
```

In an authoritative base game, no port field is unknown. In ingestion/staging,
use explicit `UNKNOWN`; never infer generic from null.

### `PerspectiveObservation/v1` — current decision

The observation should contain:

- atlas/setup IDs and hashes;
- observer identity and relative-seat aliases;
- public board occupancy and robber;
- public player statistics and visible achievements;
- observer's exact hand and development cards;
- opponents' public card totals and, if maintained, explicitly labeled belief
  ranges/distributions;
- current phase/prompt/actor;
- active trades and perspective-visible event delta;
- complete legal action candidates.

Keep these epistemic classes distinct:

```text
AUTHORITATIVE_PUBLIC
AUTHORITATIVE_PRIVATE_SELF
PUBLICLY_DERIVED
MODEL_BELIEF
```

Opponent exact hands and hidden development-card identities belong only to the
omniscient engine state and must not cross this boundary.

### Sparse text, dense numeric arrays

The typed contract can be dense for validation. Its text renderer should be
sparse:

- always list all 19 tile and nine port assignments;
- list only occupied corners and road slots;
- explicitly say unlisted corner/road slots are empty;
- list robber exactly once;
- keep sections and entity ordering fixed.

The graph/tensor renderer can use dense fixed arrays.

## Recommended neural graph

Treat Catan as a typed incidence graph (equivalently, a small cell complex):

- 19 tile entities;
- 54 corner entities;
- 72 road-slot entities;
- nine port entities;
- one player entity per seat;
- one global game entity.

Typed relations:

- tile ↔ corner;
- tile ↔ road slot;
- road slot ↔ its two corners;
- port ↔ coastal road;
- port ↔ its two corners;
- player ↔ owned pieces, or owner as a dynamic categorical feature.

Raw dynamic features:

- tile: resource, exact number, pips/probability, robber;
- corner: empty/settlement/city and owner;
- road slot: empty/owner;
- port: present, ratio, resource/generic;
- player: relative seat, public VP, pieces left, public hand/dev totals, played
  cards, awards; exact private features only for self;
- global: phase, actor, roll state, bank, turn, active trades.

Keep raw and derived fields separate. For example, preserve exact `number=6`
and add `pips=5`; do not replace the number with probability.

A graph transformer can use type embeddings, static coordinate embeddings, and
an incidence/shortest-path attention bias. A simpler message-passing network is
also a valid baseline. The value head pools global/player/entity states.

## Coordinates, IDs, descriptors, and symmetry

### Tile coordinates

Keep cube coordinates in the atlas because neighbor arithmetic and rotations are
simple. An axial `(q,r)` rendering may save prompt tokens, but coordinates need
not be repeated in ordinary decisions once the atlas is fixed.

### Corner coordinates

Use canonical corner IDs as the primary handle. If coordinates are exposed,
freeze the existing unique integer triple as a separate named system such as
`vertex_cube3/v1`. Do not imply it is a land-tile cube coordinate.

### Road coordinates

Use sorted endpoint identity: `E19_46`. This is both canonical and
self-describing. A separate sequential edge index adds little value except for a
dense tensor index table.

### Reasoning descriptors

A canonical ID alone is arbitrary to an untuned model. Pair it with a dynamic,
human-readable descriptor:

```text
N01 [T00:11-WOOD | T01:6-SHEEP | T06:5-WOOD; 11 pips]
N25 [T07:11-SHEEP; 2 pips; P00 2:1-BRICK]
P00 [2:1 BRICK on E25_26; corners N25,N26]
```

Descriptors are aliases, not identities. They can collide, and they change when
a new board is generated. The action still targets `N01`, not “the 6-5-11
spot.”

### Actual base-atlas symmetry

The land hex/corner/road topology has sixfold dihedral symmetry. The **fixed
presence pattern of nine alternating port slots breaks this**.

An exhaustive check of the 12 cube-coordinate rotations/reflections against the
repository's typed `Land/Port/Water` topology found only:

- rotations 0°, 120°, and 240°;
- three corresponding reflections.

So the full base atlas has a `D3` automorphism group of six transforms, not full
`D6`. A 60° rotation maps fixed port hexes to non-port water hexes.

Implications:

- use the six true atlas automorphisms for exact state/action augmentation;
- transform tile setup, port types, pieces, robber, and action targets together;
- do not apply arbitrary 60° rotations unless the schema is expanded to 18
  optional coastal port positions or the transformed board is intentionally
  treated as out-of-distribution;
- do not rotate transcript-grounded examples containing “top,” “left,” or
  similar screen-direction evidence unless the language is transformed too.

For a GNN, also test arbitrary internal node-array permutations while preserving
incidence; output logits should permute equivariantly. For an atlas-token LLM,
random ID relabeling is not a desired production symmetry because token identity
is deliberately tied to a fixed location.

## Legal actions and model output

Representation and legality should be related but separate contracts.

For a text model, provide only the current legal candidates:

```text
LEGAL  # choose one index; every row is engine-validated
0 SETTLE N00 [11-WOOD / 4-WOOD / 5-WOOD; 9 pips]
1 SETTLE N01 [11-WOOD / 6-SHEEP / 5-WOOD; 11 pips]
...
```

Each record needs:

- ephemeral packet index;
- stable canonical action fingerprint, for example
  `BUILD_SETTLEMENT@N01`;
- action type;
- canonical targets/payload;
- deterministic factual descriptor.

Store training labels by canonical action fingerprint. Derive the menu index at
packet construction time. Never store only an index whose meaning depends on
menu order.

For a learned graph/entity policy, score candidates rather than forcing the
model to output raw coordinates:

```text
score(action) = MLP(global, actor, action_type, target_embeddings, payload)
```

Normalize over the legal set. This naturally handles a variable number of
settlement, road, and robber candidates. Trades and other combinatorial payloads
can use conditional/factorized heads, following the Catan practitioner work and
AlphaStar precedent.

The engine validates the chosen fingerprint/index and executes the associated
action. The policy never gets a “free-form geometry” escape hatch.

## Measured format costs in this repository

### Method

One representative setup state was used:

- contract: `evals/catan_board_bench/datasets/catan_board_bench_100/contracts/sample_000.json`;
- replay game `191035308`, step 8;
- BLUE's second-settlement decision;
- 40 legal settlement actions;
- tokenizers: locally cached Qwen2.5-7B-Instruct and Qwen3-8B tokenizers.

The two tokenizer snapshots available locally gave identical counts for these
samples. These are exploratory measurements from this research session, not a
committed reproducibility artifact; implementation should add a script that pins
model revisions and saves the exact rendered inputs and tokenizer hashes.

### Results

```text
Format                                      Characters   Tokens
Current full pretty CatanBoardBench JSON             73,900   23,547
Same full contract, minified                    34,535   14,989
Compact self-contained topology + state          3,571    2,418
Compact atlas-known JSON                         1,313      632
Compact atlas-known section DSL                    849      446
Natural-language board delta                     1,515      562
Current replay observation only                  1,000      419
Current verbose legal menu only                  6,162    2,205
Current complete decision prompts                9,554    3,145
```

A prototype full-word compact legal menu used 2,020 characters / 1,325 base
Qwen tokens. Combining it with an atlas-token DSL used:

- 1,814 tokens with the unmodified Qwen3 tokenizer;
- 1,456 tokens after adding the repository's 197 Catan tokens.

For comparison, current observation plus current legal menu was 2,624 tokens in
this state.

Caveats:

- this is one setup state, not a distribution;
- token count does not measure comprehension accuracy;
- the complete prompt includes instructions beyond board representation;
- the counts are not yet backed by a committed measurement script or manifest;
- the atlas-known forms assume either a learned/cached atlas or sufficiently
  self-describing legal candidates;
- main-game private state, trades, and event history add tokens to every format.

The measurement establishes only that the current full contract is unsuitable
as prompt text and that a compact sectioned format is plausibly affordable.

## Controlled representation bake-off

Do not freeze the model-facing renderer from literature or taste alone.

### Data

Sample decision states by game lineage from held-out replays, stratified across:

- first and second setup placements;
- road placement and expansion;
- robber moves;
- city/settlement/build decisions;
- port access and maritime trades;
- dense late-game road networks;
- all port resources and generic ports;
- ambiguous number/resource descriptors.

Keep CatanBoardBench-100 evaluation-only and use non-overlapping games for format
development.

### Teacher-model input conditions

Compare at least:

1. current replay packet;
2. compact canonical JSON + indexed legal menu;
3. sectioned Catan DSL + indexed legal menu;
4. DSL + self-describing target descriptors;
5. image only;
6. image + DSL.

Use plain IDs for untuned APIs. Hold system instructions, history, private state,
legal candidates, temperature, and output schema constant.

### Student conditions

Compare:

1. DSL with ordinary tokenizer;
2. DSL with trained atomic Catan tokens;
3. Catanatron-style mixed tensor baseline;
4. heterogeneous graph/entity encoder + candidate scorer;
5. optional graph/entity prefix projected into the language model.

Match training examples, split, and approximately match head capacity where
possible.

### Separate decoding from strategy

Representation comprehension tests:

- reconstruct all tile assignments;
- reconstruct all port assignments and attachments;
- node/edge occupancy;
- robber location;
- adjacency and reachability;
- legal-target membership;
- full public-board fingerprint.

Decision tests:

- selected action is parseable and in the exact legal set;
- expert-action top-1/top-k imitation;
- engine rollout or value estimate under common random numbers;
- eventual self-play strength.

Do not infer that a format is strategically good merely because it is easy to
parse, and do not blame strategy when the model decoded the board incorrectly.

### Robustness tests

- swap one tile attribute; only dependent answers should change;
- swap one port type while keeping its slot fixed;
- move one robber/piece/road;
- apply all six valid `D3` atlas automorphisms;
- reorder equivalent text sections/records;
- vary JSON/DSL syntax without changing content;
- for GNNs, permute array indexing while preserving graph incidence;
- test descriptors that collide and require canonical IDs to disambiguate;
- remove the image to compute visual gain.

### Metrics

- schema/parse validity;
- exact field and full-contract accuracy;
- illegal-target rate, with a required retained-data target of zero;
- action top-1/top-k;
- target-flip accuracy and unchanged-fact specificity;
- symmetry consistency/equivariance;
- prompt-order variance;
- input tokens, cached/uncached cost, and latency;
- student sample efficiency and self-play Elo/win rate.

## Recommended implementation order after schema approval

1. Generate and freeze `BoardAtlas/v1` from the engine, with a topology hash and
   golden invariants.
2. Add strict `BoardSetup/v1` and `PerspectiveObservation/v1` objects; no model
   renderer changes yet.
3. Implement round-trip fingerprints and privacy/namespace/port validation.
4. Add two pure adapters: compact JSON and sectioned DSL.
5. Add canonical dynamic descriptors and stable action fingerprints.
6. Run the teacher format bake-off before replacing the shared replay/live
   formatter.
7. Train the text student on the winning renderer and atomic atlas tokens.
8. Build the heterogeneous graph/tensor baselines only after the textual
   contract is stable, so all models are compared from identical information.

## Non-negotiable invariants

- Engine state is the oracle; images and prose are views.
- Atlas IDs and hashes are versioned data/training contracts.
- Port slot and port trade type are different fields with different lifetimes.
- `DESERT`, `GENERIC_PORT`, `ABSENT`, and `UNKNOWN` are distinct values.
- Every source-specific ID is namespaced.
- Every model observation is perspective-safe.
- Every candidate action is engine-generated and exactly targetable.
- Model outputs select a candidate; they do not invent coordinates.
- Raw state and derived features such as pips remain distinguishable.
- A model-facing format is accepted only after exact decoding, robustness,
  token-cost, and decision ablations.

## Bottom line

The most intuitive robust representation is not a single coordinate system. It
is a **stable atlas plus changing attributes**, rendered with both exact IDs and
Catan-language aliases.

For the user's concrete port concern:

```text
P00 / E25_26 / N25,N26     # permanent geometry
P00 = 2:1 BRICK            # one game's setup value
BLUE owns N25              # one decision's dynamic occupancy
BLUE therefore has 2:1 BRICK access  # deterministic derived fact
```

That decomposition is exact, easy to validate, natural to a model, and usable
by text LLMs, graph policies, tensors, replay tools, and visual training without
changing the underlying meaning.
