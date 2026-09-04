# Native vision-SFT infrastructure fixture

This four-row fixture is only for a one-step architecture/save/reload smoke. It
uses the four tracked images in `../modal_vlm_smoke/images/` and retains only
short atomic node-occupancy targets:

```text
EMPTY
<RED> <SETTLEMENT>
<RED> <CITY>
<BLUE> <SETTLEMENT>
```

It intentionally excludes the long board-bbox and local-state serialization
rows in the general Modal fixture. It is not a train/eval split and must never be
reported as model-quality evidence.
