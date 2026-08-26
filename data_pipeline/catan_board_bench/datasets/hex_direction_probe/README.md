# Hex Direction Probe

A balanced visual diagnostic for one-hop pointy-top hex directions. Every candidate label occupies every direction exactly once. The two question types test direction-to-label and label-to-direction mapping. No board atlas or contract is supplied.

For a full balanced probe, run the evaluator with `--suite probe`, both hex categories, and `--limit-samples 0`; positive limits are applied per category and may select an unbalanced prefix.
