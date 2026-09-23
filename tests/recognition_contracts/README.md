# Board-recognition reorganization contracts

Run with the project's selected interpreter:

```sh
uv run --no-sync python -m pytest -q tests/recognition_contracts
```

The local `.venv/bin/pytest` launcher currently has a stale shebang pointing at
another checkout. Its Pillow runtime produces different PNG compression bytes;
`python -m pytest` uses the same interpreter as the pre-edit witnesses.

`test_parity.py` pins witnesses captured before the cleanup: complete inverse
rows, pair placements, query scheduling, semantic rows, density ordering,
production padding, spatial targets, relation rows, marker/probe rows, 292 PNG
files, and fixture dataset queries. Do not refresh these hashes to hide drift.

`reference.py` reads the inspected clean source revision from Git. Full export
tests run original and reorganized implementations at the same temporary path
and compare every output byte. These checks need the existing replay_v1 corpus
and take several minutes. Pair-rendering checks use the existing curriculum
fixture and compare both serial and spawned-worker output with the original.

Package initializers retain the original API, exceptions, constants, and
monkeypatch entry points. Extracted operations look up shared dependencies on
that API at call time; imports themselves stay top-level. Re-exported functions
are the implementation objects, without wrappers. The dataset class remains
defined at its original module name for pickle compatibility, and its project
root accounts for the added directory level.

Seven physical source files remain at the root because external SFT generators
open and hash those exact paths: full_board_readout, node_edge_readout,
replay_dataset, single_piece_localization, sources, spatial_robber, and
terrain_readout. `reference.py` lists them as `HASHED_SOURCES`, and
`test_package_root_and_source_hash_paths` asserts each is still a file and not a
directory. The readers are `build_symbolic_board_dataset/_sources.py`,
`build_spatial_continuation_dataset/_sources.py` and
`build_board_fluency_review/_build.py`; each hashes the file's current bytes per
run, so editing a file only changes future manifests, but splitting one into a
package breaks the open. Their migration needs a matching change to those three
SFT path lists. Historical manifests and hash validators are unchanged.
