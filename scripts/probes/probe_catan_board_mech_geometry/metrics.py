"""Leave-one-out identity decoding and topology neighbour retrieval."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cle.players.data import JsonValue
from scripts.probes.probe_catan_board_mech_geometry.model_io import LayerVectors

JsonDict = dict[str, JsonValue]
FloatArray = npt.NDArray[np.float32]

__all__ = ["FloatArray", "JsonDict", "evaluate_identity_by_layer", "evaluate_topology_by_layer"]


def evaluate_identity_by_layer(
    layer_data: LayerVectors,
    min_samples_per_class: int,
) -> JsonDict:
    summary: JsonDict = {}
    for layer_idx, samples in sorted(layer_data.items()):
        if not samples:
            summary[str(layer_idx)] = {"samples": 0}
            continue

        labels = np.array([int(y) for _, y in samples], dtype=np.int32)
        feats = np.stack([vec.numpy() for vec, _ in samples], axis=0).astype(np.float32)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)

        cls, counts = np.unique(labels, return_counts=True)
        class_counts = {int(c): int(n) for c, n in zip(cls.tolist(), counts.tolist())}
        usable = [c for c, n in class_counts.items() if n >= min_samples_per_class]
        if not usable:
            summary[str(layer_idx)] = {
                "samples": int(len(labels)),
                "classes": int(len(cls)),
                "usable_classes": 0,
                "id_loo_accuracy": 0.0,
                "tested": 0,
                "note": "too_few_per_class",
            }
            continue

        mask = np.isin(labels, usable)
        labels = labels[mask]
        feats = feats[mask]
        if len(labels) == 0:
            summary[str(layer_idx)] = {
                "samples": 0,
                "classes": int(len(cls)),
                "usable_classes": 0,
            }
            continue

        sum_by_class: dict[int, FloatArray] = {}
        count_by_class: dict[int, int] = {}
        for c in usable:
            idx = np.where(labels == c)[0]
            if len(idx):
                sum_by_class[c] = feats[idx].sum(axis=0)
                count_by_class[c] = int(len(idx))

        means = {c: sum_by_class[c] / count_by_class[c] for c in count_by_class}
        correct = 0
        tested = 0

        for sample_vec, lbl in zip(feats, labels):
            if lbl not in means or count_by_class[lbl] <= 1:
                continue
            test_means = dict(means)
            test_means[lbl] = (sum_by_class[lbl] - sample_vec) / (count_by_class[lbl] - 1)
            c_ids = sorted(test_means)
            centroid_matrix = np.stack([test_means[c] for c in c_ids], axis=0)
            centroid_matrix = centroid_matrix / (
                np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-9
            )
            pred_idx = int(np.argmax(sample_vec @ centroid_matrix.T))
            pred = c_ids[pred_idx]
            tested += 1
            if pred == int(lbl):
                correct += 1

        summary[str(layer_idx)] = {
            "samples": int(len(labels)),
            "classes": int(len(set(labels.tolist()))),
            "usable_classes": int(len(usable)),
            "id_loo_accuracy": float(correct / tested) if tested else 0.0,
            "tested": int(tested),
            "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
        }

    return summary


def evaluate_topology_by_layer(
    layer_data: LayerVectors,
    graph_neighbors: dict[int, set[int]],
    min_samples_per_class: int,
    topk_values: list[int],
) -> JsonDict:
    summary: JsonDict = {}
    for layer_idx, samples in sorted(layer_data.items()):
        if not samples:
            summary[str(layer_idx)] = {"samples": 0}
            continue

        labels = np.array([int(y) for _, y in samples], dtype=np.int32)
        feats = np.stack([vec.numpy() for vec, _ in samples], axis=0).astype(np.float32)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)

        cls, counts = np.unique(labels, return_counts=True)
        class_counts = {int(c): int(n) for c, n in zip(cls.tolist(), counts.tolist())}
        usable = [c for c, n in class_counts.items() if n >= min_samples_per_class]
        if len(usable) < 2:
            summary[str(layer_idx)] = {
                "samples": int(len(labels)),
                "classes": int(len(set(labels.tolist()))),
                "usable_classes": int(len(usable)),
                "note": "too_few_classes",
            }
            continue

        means: dict[int, FloatArray] = {}
        for c in usable:
            idx = np.where(labels == c)[0]
            means[c] = feats[idx].mean(axis=0)

        ids = sorted(means)
        mean_matrix = np.stack([means[c] for c in ids], axis=0)
        normed = mean_matrix / (np.linalg.norm(mean_matrix, axis=1, keepdims=True) + 1e-9)
        sim = normed @ normed.T

        metric_out: JsonDict = {}
        for k in topk_values:
            precision_scores: list[float] = []
            recall_scores: list[float] = []
            mrr_scores: list[float] = []
            for i, node_id in enumerate(ids):
                true_neighbors = graph_neighbors.get(node_id, set())
                if not true_neighbors:
                    continue
                row = sim[i].copy()
                row[i] = -2.0
                order = np.argsort(-row)
                pred = [ids[j] for j in order[:k]]
                hit_set = set(pred) & set(true_neighbors)
                precision_scores.append(len(hit_set) / k)
                recall_scores.append(len(hit_set) / len(true_neighbors))
                for rank, j in enumerate(order, start=1):
                    if ids[j] in true_neighbors:
                        mrr_scores.append(1.0 / rank)
                        break

            metric_out[f"top_{k}"] = {
                "precision": float(np.mean(precision_scores)) if precision_scores else 0.0,
                "recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
                "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
                "nodes_with_neighbors": int(len(precision_scores)),
            }

        summary[str(layer_idx)] = {
            "samples": int(len(labels)),
            "classes": int(len(set(labels.tolist()))),
            "usable_classes": int(len(usable)),
            "metrics": metric_out,
        }

    return summary
