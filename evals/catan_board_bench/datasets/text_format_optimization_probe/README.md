# Catan Text-Format Optimization Probe

A strict comparison between the incumbent `tile_rows` representation and three
query-indexed, lossless projections of the same complete public graph.

The derived indexes expose named tile neighbors, tile-corner state, port-endpoint
state, roll-to-tile/source lookup, and player-owned entity-ID lists. They do not
expose player entity counts, aggregated roll payouts, or any question-specific
answer.
