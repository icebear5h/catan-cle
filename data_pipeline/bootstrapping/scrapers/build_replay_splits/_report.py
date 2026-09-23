"""Render the human-readable markdown companion to the split manifest."""

import json
from pathlib import Path

from data_pipeline.bootstrapping.scrapers.build_replay_splits._models import Manifest


def write_markdown(path: Path, manifest: Manifest) -> None:
    """Write the split counts, colour balance, and step plans as markdown."""

    lines = [
        "# Colonist Replay Splits",
        "",
        f"Generated: {manifest['generated_at']}",
        f"Total games: {manifest['summary']['total_games']}",
        "",
        "## Split Counts",
        "",
        "| Split | Games | Raw replays present |",
        "| --- | ---: | ---: |",
    ]
    for split, data in manifest["summary"]["splits"].items():
        lines.append(f"| {split} | {data['games']} | {data['raw_replays_present']} |")

    lines.extend(["", "## Balance Color Counts", ""])
    for split, data in manifest["summary"]["splits"].items():
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| Color | Games |")
        lines.append("| --- | ---: |")
        for color, count in data["balance_color_counts"].items():
            lines.append(f"| {color} | {count} |")
        lines.append("")

    lines.extend(
        [
            "## Step Plans",
            "",
            "`early_mid` is the current training target. `late_hard` is kept separate for a harder later dataset.",
            "",
            "```json",
            json.dumps(manifest["step_plans"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines))


__all__ = ["write_markdown"]
