"""Precision / recall / F1 exactly as defined in the brief.

    Precision = |correct matches| / |all system matches|
    Recall    = |correct matches| / |all transactions that should match|
    F1        = 2PR / (P + R)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass
class Scores:
    precision: float
    recall: float
    f1: float
    tp: int
    predicted: int
    expected: int

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} "
                f"(correct {self.tp} / proposed {self.predicted} / expected {self.expected})")


def evaluate(predicted: Iterable[tuple[str, str]], truth: Iterable[tuple[str, str]]) -> Scores:
    pred, gold = set(predicted), set(truth)
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return Scores(p, r, f1, tp, len(pred), len(gold))
