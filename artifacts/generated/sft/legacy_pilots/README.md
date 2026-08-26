# Quarantined Legacy Pilots

These May 2026 subsets are preserved for provenance, not as valid train/eval
splits. In particular, `post_atlas_train_short_100.jsonl` and
`post_atlas_heldout_short_100.jsonl` share 12 source images. Do not report the
held-out file as an independent visual evaluation set.

Regenerate future splits by grouping on source sample/image before assigning
rows to train, validation, or test partitions.
