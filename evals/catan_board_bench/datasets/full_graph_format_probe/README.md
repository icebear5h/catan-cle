# Catan Full-Graph Format Probe

A strict, text-only comparison of six lossless renderings of the same public
Catan board facts and the same 60 diagnostic questions.

Formats:

- `optimized_html`
- `full_graph_json`
- `datalog`
- `sql_relational`
- `integrated_ascii`
- `tile_rows` (byte-identical incumbent baseline)

Every rendering round-trips to the source `catan_full_public_graph/v1` digest.
The compact non-baseline formats may omit redundant incidence lists only when
they can be deterministically reconstructed from tile, edge, and port topology.
No representation contains player summaries or precomputed question answers.

`integrated_ascii` places tile state, buildings, and roads directly in the
global board diagram. Its static appendix contains topology and ports but does
not repeat dynamic tile, node, or edge state.
