"""Command-line interface: ``python -m recon <command>``.

    reconcile   run the full match -> review -> improve cycle on one batch
    explain     show what the model has learned (term alignments, lag/amount profile)
    generate    write a synthetic multi-month dataset
    simulate    month-by-month simulation with an oracle reviewer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import id_ground_truth, load_bank, load_register
from .matcher import MatcherConfig, MLMatcher
from .svd_model import SVDConfig
from .unique import UniqueConfig
from .workflow import (InteractiveReviewer, OracleReviewer, ReconciliationSession, ThresholdReviewer,
                       TrainingStore, no_resolver, oracle_resolver)


def _load_truth(arg: str | None, bank, reg) -> set[tuple[str, str]] | None:
    if not arg:
        return None
    if arg == "id":
        return id_ground_truth(bank, reg)
    df = pd.read_csv(arg)
    return set(zip(df["bank_id"].astype(str), df["register_id"].astype(str)))


def _interactive_resolver(bank_left, reg_left, suggestions):
    """Manual mop-up in the terminal: type 'BANK_ID REG_ID' pairs, blank line to finish."""
    if len(bank_left) == 0 or len(reg_left) == 0:
        return []
    print("\nUnmatched bank:")
    print(bank_left[["id", "date", "amount", "type", "description"]].to_string(index=False))
    print("\nUnmatched register:")
    print(reg_left[["id", "date", "amount", "type", "text"]].to_string(index=False))
    if suggestions:
        print("\nLow-confidence suggestions:")
        for s in suggestions:
            print(f"  {s.bank_id} <-> {s.reg_id}  conf={s.confidence:.2f}")
    pairs, bl, rl = [], set(bank_left["id"]), set(reg_left["id"])
    while True:
        line = input("pair (BANK_ID REG_ID), blank to finish: ").strip()
        if not line:
            return pairs
        parts = line.split()
        if len(parts) == 2 and parts[0] in bl and parts[1] in rl:
            pairs.append((parts[0], parts[1]))
            bl.discard(parts[0])
            rl.discard(parts[1])
        else:
            print("  not a valid unmatched pair")


def _matcher_cfg(args) -> MatcherConfig:
    return MatcherConfig(svd=SVDConfig(k=args.components), accept_threshold=args.threshold)


def cmd_reconcile(args) -> int:
    bank, reg = load_bank(args.bank), load_register(args.register)
    truth = _load_truth(args.truth, bank, reg)
    store = TrainingStore.load(args.state) if args.state else TrainingStore()
    print(f"Loaded {len(bank)} bank / {len(reg)} register transactions; {len(store)} validated pairs in training store")

    if args.review == "oracle" or args.manual == "oracle":
        if truth is None:
            print("--review/--manual oracle needs --truth", file=sys.stderr)
            return 2
    if args.review == "interactive":
        reviewer = InteractiveReviewer()
    elif args.review == "oracle":
        reviewer = OracleReviewer(truth, error_rate=args.reviewer_error)
    else:
        reviewer = ThresholdReviewer(float(args.review.split(":")[1]) if ":" in args.review else 0.9)
    resolver = {"oracle": oracle_resolver(truth) if truth else None, "none": no_resolver,
                "interactive": _interactive_resolver}[args.manual]

    session = ReconciliationSession(bank, reg, store, UniqueConfig(max_lag_days=args.max_lag),
                                    _matcher_cfg(args))
    session.run_cycle(reviewer, resolver, log=print)

    report = session.report()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.out, index=False)
        print(f"Match report -> {args.out}")
    flagged = report[report["flags"] != ""]
    if len(flagged):
        print(f"\nFlagged for attention ({len(flagged)}):")
        cols = ["bank_id", "register_id", "stage", "status", "confidence", "flags", "bank_description", "register_description"]
        print(flagged[cols].head(25).to_string(index=False))
    if args.state:
        Path(args.state).parent.mkdir(parents=True, exist_ok=True)
        store.save(args.state)
        print(f"\nTraining store ({len(store)} pairs) -> {args.state}")
    if truth is not None:
        print("\nEvaluation against ground truth:")
        for name, sc in session.scores(truth).items():
            print(f"  {name:<15} {sc}")
    return 0


def cmd_explain(args) -> int:
    store = TrainingStore.load(args.state)
    if not len(store):
        print("training store is empty - run `reconcile --state ...` first")
        return 1
    m = MLMatcher().fit(store.pairs)
    print(f"{len(store)} validated pairs ({pd.Series([p.source for p in store.pairs]).value_counts().to_dict()})\n")
    if m.model is not None:
        print("Strongest learned word alignments (bank term -> register term, normalised MI):")
        for a, b, w in m.model.top_alignments(args.top):
            print(f"  {a.split(':', 2)[2]:>14}  ->  {b.split(':', 2)[2]:<14} {w:.2f}")
        print("\nType-code alignment:", [(a.split(':')[2], b.split(':')[2], round(w, 2)) for a, b, w in m.model.top_alignments(5, 't')])
    lags = np.array([p.lag for p in store.pairs])
    print(f"\nLearned date lag (bank - register): median {np.median(lags):.0f}d, 95% within "
          f"[{np.percentile(lags, 2.5):.0f}, {np.percentile(lags, 97.5):.0f}]d")
    diffs = np.array([p.amount_diff_cents for p in store.pairs])
    print(f"Amount discrepancies: {(diffs > 0).mean():.1%} of pairs differ, max {diffs.max():.0f} cents")
    return 0


def cmd_generate(args) -> int:
    from .synth import SynthConfig, generate
    b, r, t = generate(SynthConfig(months=args.months, seed=args.seed, volume=args.volume,
                                   drift_month=args.drift_month or None))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    b.to_csv(out / "bank_statements.csv", index=False)
    r.to_csv(out / "check_register.csv", index=False)
    pd.DataFrame(sorted(t), columns=["bank_id", "register_id"]).to_csv(out / "truth.csv", index=False)
    print(f"{len(b)} bank, {len(r)} register, {len(t)} true pairs -> {out}/")
    return 0


def cmd_simulate(args) -> int:
    from .simulate import SimConfig, simulate
    b, r = pd.read_csv(args.bank), pd.read_csv(args.register)
    t = pd.read_csv(args.truth)
    truth = set(zip(t["bank_id"], t["register_id"]))
    df = simulate(b, r, truth, _matcher_cfg(args),
                  SimConfig(carry_training=not args.no_carry, reviewer_error=args.reviewer_error))
    cols = ["month", "true_pairs", "train_pairs_before", "unique_proposed", "ml_candidates", "ml_proposed",
            "ml_precision", "ml_recall", "ml_f1", "auto_precision", "auto_recall"]
    print(df[cols].round(3).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recon", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def model_opts(sp):
        sp.add_argument("--components", type=int, default=100, help="SVD principal components (default 100)")
        sp.add_argument("--threshold", type=float, default=0.6, help="min confidence for an ML proposal (default 0.6)")
        sp.add_argument("--reviewer-error", type=float, default=0.0, help="oracle reviewer error rate")

    r = sub.add_parser("reconcile", help="run match -> review -> improve on one batch")
    r.add_argument("--bank", default="data/bank_statements.csv")
    r.add_argument("--register", default="data/check_register.csv")
    r.add_argument("--state", help="training-store JSON; loaded before and saved after (enables improvement across runs)")
    r.add_argument("--review", default="interactive", help="interactive | oracle | auto[:min_conf]")
    r.add_argument("--manual", default="none", choices=["none", "oracle", "interactive"])
    r.add_argument("--truth", help="'id' (shared id suffix, provided data) or a CSV with bank_id,register_id")
    r.add_argument("--out", help="write the match report CSV here")
    r.add_argument("--max-lag", type=int, default=5, help="expected max register->bank lag for unique-stage flags")
    model_opts(r)
    r.set_defaults(func=cmd_reconcile)

    e = sub.add_parser("explain", help="show what the model has learned")
    e.add_argument("--state", required=True)
    e.add_argument("--top", type=int, default=20)
    e.set_defaults(func=cmd_explain)

    g = sub.add_parser("generate", help="write a synthetic multi-month dataset")
    g.add_argument("--out", default="data/synthetic")
    g.add_argument("--months", type=int, default=12)
    g.add_argument("--volume", type=float, default=2.0)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--drift-month", type=int, default=7, help="month the bank changes format (0 = never)")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("simulate", help="month-by-month simulation with an oracle reviewer")
    s.add_argument("--bank", default="data/synthetic/bank_statements.csv")
    s.add_argument("--register", default="data/synthetic/check_register.csv")
    s.add_argument("--truth", default="data/synthetic/truth.csv")
    s.add_argument("--no-carry", action="store_true", help="forget training data between months (baseline)")
    s.add_argument("--out")
    model_opts(s)
    s.set_defaults(func=cmd_simulate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
