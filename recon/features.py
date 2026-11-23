"""Turning a transaction into a bag of "words".

Following the paper, each transaction is a *sentence* in its source's
*language*. Every term is prefixed with its source (``b:`` / ``r:``) so the
bank's ``"gas"`` and the register's ``"gas"`` are different words, exactly like
cognates in two languages: the model has to *learn* that they align.

Term families (all optional, controlled by :class:`FeatureConfig`):

* ``w:<token>``  - description words, digits collapsed to ``<num>``
* ``g:<ngram>``  - character trigrams of each word (robust to typos such as
  "Groceres" / "Amazon rder" and to store numbers glued to words)
* ``t:<type>``   - raw transaction-type code (DEBIT, DR, ...)
* ``a:<bucket>`` - coarse log2 amount magnitude (payroll is "big", fees are "small")

Dates are deliberately *not* words: as the paper notes, a date from one month
says nothing about the next. Date information enters through the learned lag
distribution in :mod:`recon.scoring` instead.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class FeatureConfig:
    words: bool = True
    char_ngrams: bool = True
    ngram_n: int = 3
    type_terms: bool = True
    amount_terms: bool = True


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens; pure numbers become ``<num>``."""
    toks = []
    for tok in _TOKEN_RE.findall(text.lower()):
        toks.append("<num>" if tok.isdigit() else tok)
    return toks


def char_ngrams(token: str, n: int = 3) -> list[str]:
    """Character n-grams with word-boundary markers: ``gas`` -> ``^ga, gas, as$``."""
    if token == "<num>":
        return []
    padded = f"^{token}$"
    if len(padded) <= n:
        return [padded]
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def amount_bucket(amount: float) -> int:
    return int(math.floor(math.log2(max(abs(amount), 1.0))))


def transaction_terms(text: str, ttype: str, amount: float, side: str, cfg: FeatureConfig) -> list[str]:
    """All terms for one transaction. ``side`` is ``"b"`` (bank) or ``"r"`` (register)."""
    terms: list[str] = []
    toks = tokenize(text)
    if cfg.words:
        terms += [f"{side}:w:{t}" for t in toks]
    if cfg.char_ngrams:
        for t in toks:
            terms += [f"{side}:g:{g}" for g in char_ngrams(t, cfg.ngram_n)]
    if cfg.type_terms and ttype:
        terms.append(f"{side}:t:{ttype.lower()}")
    if cfg.amount_terms:
        terms.append(f"{side}:a:{amount_bucket(amount)}")
    return terms


def frame_terms(df, side: str, cfg: FeatureConfig) -> list[list[str]]:
    """Terms for every row of a canonical frame (see :mod:`recon.data`)."""
    return [
        transaction_terms(t, ty, a, side, cfg)
        for t, ty, a in zip(df["text"], df["type"], df["amount"])
    ]
