"""The human-in-the-loop monthly cycle (paper Figure 4).

    unique-amount match -> QC -> ML ("cross-language") match -> QC -> manual mop-up
                                     ^                                   |
                                     +------- validated pairs -----------+

Only matches a human made or accepted enter the :class:`TrainingStore`, so QC
is both a quality gate *and* the training signal. The store is persisted as
JSON and carried from one batch (month) to the next - this is how the system
"improves as more validated matches are added".
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .matcher import MatcherConfig, MLMatcher, TrainingPair, pairs_from_frames
from .metrics import Scores, evaluate
from .unique import (AMOUNT_DIFF, LAG_EXCEEDS, LOW_CONFIDENCE, LOW_TEXT_SIM, REGISTER_AFTER_BANK,
                     SOURCE_NOTE, TYPE_MISMATCH, UniqueConfig, unique_amount_matches)


# --------------------------------------------------------------------------- store
class TrainingStore:
    """The growing parallel corpus of validated matches."""

    def __init__(self, pairs: Iterable[TrainingPair] = ()):
        self.pairs: list[TrainingPair] = list(pairs)

    def add(self, pairs: Iterable[TrainingPair]) -> None:
        self.pairs.extend(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def save(self, path: str | Path) -> None:
        rows = []
        for p in self.pairs:
            d = asdict(p)
            d["bank_date"] = p.bank_date.strftime("%Y-%m-%d")
            d["reg_date"] = p.reg_date.strftime("%Y-%m-%d")
            rows.append(d)
        Path(path).write_text(json.dumps(rows, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "TrainingStore":
        p = Path(path)
        if not p.exists():
            return cls()
        rows = json.loads(p.read_text())
        for r in rows:
            r["bank_date"] = pd.Timestamp(r["bank_date"])
            r["reg_date"] = pd.Timestamp(r["reg_date"])
        return cls(TrainingPair(**r) for r in rows)


# ------------------------------------------------------------------------- records
@dataclass
class MatchRecord:
    bank_id: str
    reg_id: str
    stage: str                      # unique | ml | manual
    confidence: float
    flags: list[str] = field(default_factory=list)
    status: str = "proposed"        # proposed | accepted | rejected
    detail: dict = field(default_factory=dict)


class Reviewer(Protocol):
    def __call__(self, record: MatchRecord, bank_row: pd.Series, reg_row: pd.Series) -> bool: ...


class OracleReviewer:
    """Simulated human QC that knows the truth (optionally makes mistakes)."""

    def __init__(self, truth: set[tuple[str, str]], error_rate: float = 0.0, seed: int = 0):
        self.truth, self.error_rate, self.rng = truth, error_rate, random.Random(seed)

    def __call__(self, record, bank_row, reg_row) -> bool:
        correct = (record.bank_id, record.reg_id) in self.truth
        return (not correct) if self.rng.random() < self.error_rate else correct


class ThresholdReviewer:
    """'Scaled-back QC': auto-accept anything at or above a confidence level."""

    def __init__(self, min_confidence: float = 0.9):
        self.min_confidence = min_confidence

    def __call__(self, record, bank_row, reg_row) -> bool:
        return record.confidence >= self.min_confidence


class InteractiveReviewer:
    """Terminal QC: accept / reject each proposal."""

    def __init__(self, input_fn: Callable[[str], str] = input, print_fn: Callable[..., None] = print):
        self.input, self.print = input_fn, print_fn
        self.accept_rest = False

    def __call__(self, record, bank_row, reg_row) -> bool:
        if self.accept_rest:
            return True
        self.print(
            f"\n[{record.stage}] conf={record.confidence:.3f} flags={','.join(record.flags) or '-'}\n"
            f"  BANK {bank_row['id']} {bank_row['date']:%Y-%m-%d} {bank_row['amount']:>10.2f} {bank_row['type']:<6} {bank_row['description']}\n"
            f"  REG  {reg_row['id']} {reg_row['date']:%Y-%m-%d} {reg_row['amount']:>10.2f} {reg_row['type']:<6} {reg_row['text']}"
        )
        while True:
            ans = self.input("  accept? [y]es / [n]o / [A]ccept all remaining: ").strip()
            if ans in ("y", "Y", ""):
                return True
            if ans in ("n", "N"):
                return False
            if ans == "A":
                self.accept_rest = True
                return True


ManualResolver = Callable[[pd.DataFrame, pd.DataFrame, list[MatchRecord]], list[tuple[str, str]]]


def oracle_resolver(truth: set[tuple[str, str]]) -> ManualResolver:
    """Simulated manual mop-up: the human finds the true pairs among what is left."""
    def resolve(bank_left, reg_left, suggestions):
        bl, rl = set(bank_left["id"]), set(reg_left["id"])
        return [(b, r) for b, r in truth if b in bl and r in rl]
    return resolve


def no_resolver(bank_left, reg_left, suggestions):
    return []


# ------------------------------------------------------------------------ session
class ReconciliationSession:
    """One batch (e.g. a month) of reconciliation."""

    def __init__(self, bank: pd.DataFrame, reg: pd.DataFrame, store: TrainingStore | None = None,
                 unique_cfg: UniqueConfig | None = None, matcher_cfg: MatcherConfig | None = None):
        self.bank, self.reg = bank.reset_index(drop=True), reg.reset_index(drop=True)
        self.store = store if store is not None else TrainingStore()
        self.unique_cfg = unique_cfg or UniqueConfig()
        self.matcher_cfg = matcher_cfg or MatcherConfig()
        self.records: list[MatchRecord] = []
        self.matcher: MLMatcher | None = None
        self._b = self.bank.set_index("id", drop=False)
        self._r = self.reg.set_index("id", drop=False)

    # -- helpers
    def accepted(self) -> list[MatchRecord]:
        return [m for m in self.records if m.status == "accepted"]

    def unmatched(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        used_b = {m.bank_id for m in self.accepted()}
        used_r = {m.reg_id for m in self.accepted()}
        return self.bank[~self.bank["id"].isin(used_b)], self.reg[~self.reg["id"].isin(used_r)]

    def _train(self) -> MLMatcher:
        self.matcher = MLMatcher(self.matcher_cfg).fit(self.store.pairs)
        return self.matcher

    # -- stage 1
    def run_unique(self) -> list[MatchRecord]:
        bank_left, reg_left = self.unmatched()
        found = unique_amount_matches(bank_left, reg_left, self.unique_cfg)
        recs = [MatchRecord(m.bank_id, m.reg_id, "unique", m.confidence, list(m.flags),
                            detail={"lag": m.lag, "chance_prob": m.chance_prob}) for m in found]
        # With a model from previous batches we can also flag descriptions that disagree
        if len(self.store) >= self.matcher_cfg.min_pairs_for_text and recs:
            self._flag_text(recs)
        self.records += recs
        return recs

    def _flag_text(self, recs: list[MatchRecord]) -> None:
        m = self.matcher or self._train()
        if m.model is None:
            return
        cut = self._cos_cut(m)
        b = self._b.loc[[r.bank_id for r in recs]]
        r_ = self._r.loc[[r.reg_id for r in recs]]
        cos = (m.model.embed(m._terms(b["text"], b["type"], b["amount"], "b")) *
               m.model.embed(m._terms(r_["text"], r_["type"], r_["amount"], "r"))).sum(1)
        for rec, c in zip(recs, cos):
            rec.detail["cosine"] = round(float(c), 3)
            if c < cut and LOW_TEXT_SIM not in rec.flags:
                rec.flags.append(LOW_TEXT_SIM)

    # -- QC
    def review(self, records: list[MatchRecord], reviewer: Reviewer) -> tuple[int, int]:
        """Apply QC; accepted pairs are added to the training store. Returns (accepted, rejected)."""
        ok = []
        for rec in records:
            if rec.status != "proposed":
                continue
            good = reviewer(rec, self._b.loc[rec.bank_id], self._r.loc[rec.reg_id])
            rec.status = "accepted" if good else "rejected"
            if good:
                ok.append((rec.bank_id, rec.reg_id))
        self.store.add(pairs_from_frames(self.bank, self.reg, ok, records[0].stage if records else "unknown"))
        self.matcher = None  # store changed -> retrain lazily
        return len(ok), len([r for r in records if r.status == "rejected"])

    # -- stage 2
    def run_ml(self) -> tuple[list[MatchRecord], list[MatchRecord]]:
        """Returns (proposals above threshold, low-confidence suggestions for the manual step)."""
        bank_left, reg_left = self.unmatched()
        # pairs rejected in QC must not be proposed again
        rejected = {(m.bank_id, m.reg_id) for m in self.records if m.status == "rejected"}
        m = self._train()
        if len(bank_left) == 0 or len(reg_left) == 0:
            return [], []
        ev = m.score(bank_left, reg_left, background=(self.bank, self.reg))
        bi = {x: i for i, x in enumerate(bank_left["id"])}
        ri = {x: j for j, x in enumerate(reg_left["id"])}
        for b, r in rejected:
            if b in bi and r in ri:
                ev["llr"][bi[b], ri[r]] = -1e6
        ev["confidence"] = m._posterior(np.maximum(ev["llr"], -50))
        rows, cols = linear_sum_assignment(-ev["llr"])

        lag_lo, lag_hi = self._lag_range(m)
        cos_cut = self._cos_cut(m)
        proposals, suggestions = [], []
        for i, j in zip(rows, cols):
            if ev["llr"][i, j] <= 0:
                # forced assignment that is more likely a non-match: leave both sides unmatched
                continue
            brow, rrow = bank_left.iloc[i], reg_left.iloc[j]
            conf = float(ev["confidence"][i, j])
            lag = int(ev["lag"][i, j])
            flags = []
            if lag < lag_lo or lag > lag_hi:
                flags.append(LAG_EXCEEDS)
            if lag < 0:
                flags.append(REGISTER_AFTER_BANK)
            if ev["amount_diff"][i, j] > 0:
                flags.append(AMOUNT_DIFF)
            if brow["direction"] and rrow["direction"] and brow["direction"] != rrow["direction"]:
                flags.append(TYPE_MISMATCH)
            if brow["notes"] or rrow["notes"]:
                flags.append(SOURCE_NOTE)
            if m.model is not None and ev["cosine"][i, j] < cos_cut:
                flags.append(LOW_TEXT_SIM)
            rec = MatchRecord(brow["id"], rrow["id"], "ml", round(conf, 4), flags, detail={
                "lag": lag, "amount_diff": round(float(ev["amount_diff"][i, j]), 2),
                "cosine": round(float(ev["cosine"][i, j]), 3), "llr": round(float(ev["llr"][i, j]), 2)})
            if conf >= self.matcher_cfg.accept_threshold:
                proposals.append(rec)
            else:
                rec.flags.append(LOW_CONFIDENCE)
                rec.status = "suggested"
                suggestions.append(rec)
        self.records += proposals + suggestions
        return proposals, suggestions

    @staticmethod
    def _lag_range(m: MLMatcher) -> tuple[int, int]:
        """Lag range covering 99% of validated matches (learned, not hard-coded)."""
        if m.lag_m.p is None:
            return 0, 5
        c = np.cumsum(m.lag_m.p)
        centers = (m.lag_m.edges[:-1] + m.lag_m.edges[1:]) / 2
        return int(centers[np.searchsorted(c, 0.005)]), int(centers[min(np.searchsorted(c, 0.995), len(c) - 1)])

    @staticmethod
    def _cos_cut(m: MLMatcher) -> float:
        if m.cos_m.p is None:
            return -1.0
        idx = np.searchsorted(np.cumsum(m.cos_m.p), 0.05)
        return float(m.cos_m.edges[idx])

    # -- stage 3
    def manual(self, resolver: ManualResolver) -> int:
        bank_left, reg_left = self.unmatched()
        suggestions = [m for m in self.records if m.status == "suggested"]
        pairs = resolver(bank_left, reg_left, suggestions)
        for b, r in pairs:
            self.records.append(MatchRecord(b, r, "manual", 1.0, status="accepted"))
        self.store.add(pairs_from_frames(self.bank, self.reg, pairs, "manual"))
        return len(pairs)

    # -- full cycle
    def run_cycle(self, reviewer: Reviewer, resolver: ManualResolver = no_resolver, log: Callable[..., None] | None = None) -> dict:
        log = log or (lambda *a, **k: None)
        u = self.run_unique()
        ua, ur = self.review(u, reviewer)
        log(f"[1] unique-amount: proposed {len(u)}, accepted {ua}, rejected {ur}")
        p, s = self.run_ml()
        ma, mr = self.review(p, reviewer)
        log(f"[2] ML matching  : proposed {len(p)}, accepted {ma}, rejected {mr}, low-confidence {len(s)} -> manual queue "
            f"(training pairs: {len(self.store)})")
        n = self.manual(resolver)
        bl, rl = self.unmatched()
        log(f"[3] manual       : resolved {n}; still unmatched bank={len(bl)} register={len(rl)}")
        return {"unique_proposed": len(u), "unique_accepted": ua, "ml_proposed": len(p), "ml_accepted": ma,
                "low_conf": len(s), "manual": n}

    # -- reporting
    def report(self) -> pd.DataFrame:
        rows = []
        for m in self.records:
            b, r = self._b.loc[m.bank_id], self._r.loc[m.reg_id]
            rows.append({
                "bank_id": m.bank_id, "register_id": m.reg_id, "stage": m.stage, "status": m.status,
                "confidence": m.confidence, "flags": ";".join(m.flags),
                "bank_date": b["date"].date(), "register_date": r["date"].date(),
                "bank_amount": b["amount"], "register_amount": r["amount"],
                "bank_description": b["description"], "register_description": r["text"],
                **{f"detail_{k}": v for k, v in m.detail.items()},
            })
        return pd.DataFrame(rows)

    def scores(self, truth: set[tuple[str, str]]) -> dict[str, Scores]:
        """System-proposed matches (before QC) scored against the truth, overall and per stage."""
        out = {}
        auto = [m for m in self.records if m.stage in ("unique", "ml") and m.status != "suggested"]
        out["automatic"] = evaluate([(m.bank_id, m.reg_id) for m in auto], truth)
        for st in ("unique", "ml"):
            recs = [(m.bank_id, m.reg_id) for m in auto if m.stage == st]
            out[st] = evaluate(recs, truth)
        out["final_after_qc"] = evaluate([(m.bank_id, m.reg_id) for m in self.accepted()], truth)
        return out
