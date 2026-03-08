from __future__ import annotations

from collections import Counter


def accuracy(gold: list[str], predicted: list[str]) -> float:
    if not gold:
        return 0.0
    correct = sum(1 for expected, actual in zip(gold, predicted) if expected == actual)
    return correct / len(gold)


def macro_f1(gold: list[str], predicted: list[str]) -> float:
    labels = sorted(set(gold) | set(predicted))
    if not labels:
        return 0.0
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, predicted) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, predicted) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, predicted) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((2 * precision * recall) / (precision + recall))
    return sum(scores) / len(scores)


def confusion(gold: list[str], predicted: list[str]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for expected, actual in zip(gold, predicted):
        matrix.setdefault(expected, Counter())
        matrix[expected][actual] += 1
    return {label: dict(counts) for label, counts in matrix.items()}

