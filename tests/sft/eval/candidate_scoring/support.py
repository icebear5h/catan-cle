"""Shared helpers for candidate ranking, neighbor confusion, and readout scoring summaries."""



ATLAS = [f"<N{i:02d}>" for i in range(54)] + ["<E00_01>", "<E00_05>"] + [f"<T{i:02d}>" for i in range(19)] + [f"<P{i:02d}>" for i in range(9)]


class _Tokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab[text]]


def _occupancy_record(
    *,
    correct: bool,
    expected: str,
    response: str,
    task_type: str = "occupancy_pair",
    partner: bool = True,
    partner_distance: int | None = 1,
    negative_distance: int | None = None,
) -> dict:
    metadata = {
        "task_type": task_type,
        "category": "node.occupancy",
        "piece": "Settlement",
        "color": "dark_blue",
    }
    if partner:
        metadata.update(
            {
                "partner_piece": "City",
                "partner_color": "red",
                "partner_distance": partner_distance,
            }
        )
    if negative_distance is not None:
        metadata["negative_distance"] = negative_distance
    return {
        "response": response,
        "metadata": metadata,
        "score": {
            "correct": correct,
            "expected_normalized": expected,
            "response_normalized": response,
        },
    }
