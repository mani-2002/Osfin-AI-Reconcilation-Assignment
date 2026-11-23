"""Month-by-month simulation of the human-in-the-loop process.

Each month: the new month's bank and register entries (plus anything left
unmatched from the previous month) go through the full cycle, with an oracle
playing the human reviewer. The training store is carried forward, which is
what lets the ML stage improve over time.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import bank_frame, register_frame
from .matcher import MatcherConfig, MLMatcher
from .metrics import evaluate
from .unique import UniqueConfig
from .workflow import OracleReviewer, ReconciliationSession, TrainingStore, oracle_resolver


@dataclass
class SimConfig:
    carry_training: bool = True      # False = forget everything each month (no learning)
    reviewer_error: float = 0.0      # probability the human QC decision is wrong
    max_carry_months: int = 1        # unmatched entries are carried at most this long


def _true_pair_cosine(store, b, r, pairs, matcher_cfg) -> float:
    """Mean SVD cosine of this month's true pairs under the model trained on *previous* months only.

    A drop signals that the "language" changed (e.g. bank format drift)."""
    m = MLMatcher(matcher_cfg or MatcherConfig()).fit(store.pairs)
    if m.model is None or not pairs:
        return float("nan")
    bi = b.set_index("id")
    ri = r.set_index("id")
    bb = bi.loc[[x for x, _ in pairs]]
    rr = ri.loc[[y for _, y in pairs]]
    eb = m.model.embed(m._terms(bb["text"], bb["type"], bb["amount"], "b"))
    er = m.model.embed(m._terms(rr["text"], rr["type"], rr["amount"], "r"))
    return float((eb * er).sum(1).mean())


def simulate(bank_raw: pd.DataFrame, reg_raw: pd.DataFrame, truth: set[tuple[str, str]],
             matcher_cfg: MatcherConfig | None = None, sim: SimConfig | None = None,
             unique_cfg: UniqueConfig | None = None) -> pd.DataFrame:
    sim = sim or SimConfig()
    bank, reg = bank_frame(bank_raw), register_frame(reg_raw)
    def month_key(d: pd.Series) -> pd.Series:
        return d.dt.year * 12 + d.dt.month - 1

    bank["period"] = month_key(bank["date"])
    reg["period"] = month_key(reg["date"])
    periods = sorted(set(bank["period"]) | set(reg["period"]))
    store = TrainingStore()
    carry_b = bank.iloc[0:0]
    carry_r = reg.iloc[0:0]
    rows = []
    for m_i, per in enumerate(periods, 1):
        b = pd.concat([carry_b, bank[bank["period"] == per]])
        r = pd.concat([carry_r, reg[reg["period"] == per]])
        if not sim.carry_training:
            store = TrainingStore()
        bids, rids = set(b["id"]), set(r["id"])
        month_truth = {(x, y) for x, y in truth if x in bids and y in rids}

        s = ReconciliationSession(b.drop(columns="period"), r.drop(columns="period"), store,
                                  unique_cfg=unique_cfg, matcher_cfg=matcher_cfg)
        n_train_before = len(store)
        prior_cos = _true_pair_cosine(store, b, r, month_truth, matcher_cfg)
        u = s.run_unique()
        s.review(u, OracleReviewer(truth, sim.reviewer_error, seed=m_i))
        acc_u = {(x.bank_id, x.reg_id) for x in u if x.status == "accepted"}
        remaining_truth = {p for p in month_truth if p not in acc_u}
        props, sugg = s.run_ml()
        s.review(props, OracleReviewer(truth, sim.reviewer_error, seed=100 + m_i))
        s.manual(oracle_resolver(truth))

        su = evaluate([(x.bank_id, x.reg_id) for x in u], month_truth)
        sm = evaluate([(x.bank_id, x.reg_id) for x in props], remaining_truth)
        sm_all = evaluate([(x.bank_id, x.reg_id) for x in props + sugg], remaining_truth)
        auto = evaluate([(x.bank_id, x.reg_id) for x in u + props], month_truth)
        rows.append({
            "month": f"{per // 12}-{per % 12 + 1:02d}", "month_idx": m_i, "bank_txns": len(b), "reg_txns": len(r),
            "true_pairs": len(month_truth), "train_pairs_before": n_train_before,
            "unique_proposed": su.predicted, "unique_precision": su.precision, "unique_recall": su.recall,
            "ml_candidates": len(remaining_truth), "ml_proposed": sm.predicted,
            "ml_precision": sm.precision, "ml_recall": sm.recall, "ml_f1": sm.f1,
            "ml_top1_precision": sm_all.precision,
            "auto_precision": auto.precision, "auto_recall": auto.recall, "auto_f1": auto.f1,
            "manual_left": len(sugg), "true_pair_cosine_prior_model": prior_cos,
        })
        # carry forward what is still unmatched (counterpart may arrive next month)
        lb, lr = s.unmatched()
        lb = lb.assign(period=month_key(lb["date"]))
        lr = lr.assign(period=month_key(lr["date"]))
        carry_b = lb[per - lb["period"] < sim.max_carry_months]
        carry_r = lr[per - lr["period"] < sim.max_carry_months]
    return pd.DataFrame(rows)
