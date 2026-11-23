"""Loading and normalising the two transaction sources.

Both sources are mapped onto one canonical frame so the rest of the system is
schema-agnostic:

    id | date | description | amount | type | direction | text | notes

* ``type`` keeps the *raw* code ("DEBIT", "DR", ...). The ML model treats it as
  just another "word", so it learns that DEBIT <-> DR from data instead of from
  a hand-written rule (the paper's robustness argument).
* ``direction`` (+1 credit / -1 debit) is derived with a small lookup and is
  used only for QC flags, never for matching decisions.
* ``text`` concatenates every free-text descriptor a source offers
  (description, plus category/notes for the register).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Only used for human-facing QC flags (TYPE_MISMATCH), not for matching.
_DIRECTION = {"DEBIT": -1, "DR": -1, "D": -1, "CREDIT": 1, "CR": 1, "C": 1}

CANONICAL_COLUMNS = ["id", "date", "description", "amount", "type", "direction", "text", "notes"]


def _canonical(df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame()
    out["id"] = df["transaction_id"].astype(str)
    out["date"] = pd.to_datetime(df["date"])
    out["description"] = df["description"].fillna("").astype(str)
    out["amount"] = df["amount"].astype(float).round(2)
    out["type"] = df["type"].fillna("").astype(str).str.upper().str.strip()
    out["direction"] = out["type"].map(_DIRECTION).fillna(0).astype(int)
    present = [c for c in text_cols if c in df.columns]
    out["text"] = df[present].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    out["notes"] = df["notes"].fillna("").astype(str) if "notes" in df.columns else ""
    return out.reset_index(drop=True)


def bank_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise a raw bank-statement frame (transaction_id, date, description, amount, type, ...)."""
    return _canonical(df, ["description"])


def register_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise a raw check-register frame (..., category, notes)."""
    return _canonical(df, ["description", "category", "notes"])


def load_bank(path: str | Path) -> pd.DataFrame:
    """Load a bank statement CSV (transaction_id, date, description, amount, type, balance)."""
    return bank_frame(pd.read_csv(path))


def load_register(path: str | Path) -> pd.DataFrame:
    """Load a check register CSV (transaction_id, date, description, amount, type, category, notes)."""
    return register_frame(pd.read_csv(path))


def id_ground_truth(bank: pd.DataFrame, register: pd.DataFrame) -> set[tuple[str, str]]:
    """Ground-truth pairs for the provided dataset.

    The synthetic files share a numeric suffix (B0047 <-> R0047). This is used
    **only** for evaluation and for simulating a human reviewer; the matcher
    never sees it.
    """
    reg_by_key = {rid[1:]: rid for rid in register["id"]}
    return {(bid, reg_by_key[bid[1:]]) for bid in bank["id"] if bid[1:] in reg_by_key}
