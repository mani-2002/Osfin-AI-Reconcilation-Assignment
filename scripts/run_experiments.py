"""Reproduce every number and figure in docs/ANALYSIS.md.

    python scripts/run_experiments.py            # all experiments (~5-10 min)
    python scripts/run_experiments.py --quick    # fewer seeds

Outputs go to results/ (CSV + PNG + summary.md).
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recon import (MatcherConfig, MLMatcher, OracleReviewer, ReconciliationSession, SVDConfig,  # noqa: E402
                   evaluate, id_ground_truth, load_bank, load_register, oracle_resolver, pairs_from_frames)
from recon.data import bank_frame, register_frame  # noqa: E402
from recon.features import FeatureConfig  # noqa: E402
from recon.simulate import SimConfig, simulate  # noqa: E402
from recon.synth import SynthConfig, generate  # noqa: E402

OUT = ROOT / "results"
# Reference categorical palette (dataviz skill, light mode), used in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

SETTINGS = {
    "full (text+date+amount)": dict(),
    "amount-blind (text+date)": dict(use_amount=False),
    "text only": dict(use_amount=False, use_date=False),
    "no text (date+amount)": dict(use_text=False, use_overlap=False),
}


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def ml_only(B, R, train_pairs, test_truth, cfg: MatcherConfig):
    """Fit on validated pairs, match the remaining transactions with the ML stage alone."""
    tb = {b for b, _ in train_pairs}
    tr = {r for _, r in train_pairs}
    m = MLMatcher(cfg).fit(pairs_from_frames(B, R, train_pairs, "gt"))
    props = m.match(B[~B["id"].isin(tb)], R[~R["id"].isin(tr)])
    above = [(p.bank_id, p.reg_id) for p in props if p.accepted_by_threshold]
    s = evaluate(above, test_truth)
    top1 = evaluate([(p.bank_id, p.reg_id) for p in props], test_truth)
    return s, top1, props


# ---------------------------------------------------------------- E1
def e1_provided(B, R, gt):
    s = ReconciliationSession(B, R)
    lines = []
    s.run_cycle(OracleReviewer(gt), oracle_resolver(gt), log=lines.append)
    sc = s.scores(gt)
    rem = gt - {(m.bank_id, m.reg_id) for m in s.records if m.stage == "unique"}
    ml = evaluate([(m.bank_id, m.reg_id) for m in s.records if m.stage == "ml" and m.status != "suggested"], rem)
    rows = [{"stage": "unique-amount", **sc["unique"].as_dict()},
            {"stage": "ML (on pairs left after unique)", **ml.as_dict()},
            {"stage": "all automatic", **sc["automatic"].as_dict()}]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e1_provided_pipeline.csv", index=False)
    s.report().to_csv(OUT / "e1_provided_matches.csv", index=False)
    return df, lines


# ---------------------------------------------------------------- E2
def e2_learning_curve(B, R, gt, seeds):
    gt_list = sorted(gt)
    sizes = [5, 10, 20, 40, 80, 160, 240]
    rows = []
    for name, kw in SETTINGS.items():
        for n in sizes:
            for seed in range(seeds):
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(gt_list), n, replace=False)
                train = [gt_list[i] for i in idx]
                test = gt - set(train)
                s, top1, _ = ml_only(B, R, train, test, MatcherConfig(seed=seed, **kw))
                rows.append({"setting": name, "train_pairs": n, "seed": seed, "precision": s.precision,
                             "recall": s.recall, "f1": s.f1, "top1_accuracy": top1.precision})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e2_learning_curve_raw.csv", index=False)
    agg = df.groupby(["setting", "train_pairs"])[["precision", "recall", "f1", "top1_accuracy"]].mean().reset_index()
    agg.to_csv(OUT / "e2_learning_curve.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    for ax, metric, title in ((axes[0], "f1", "F1 at confidence >= 0.6"), (axes[1], "top1_accuracy", "Accuracy of the 1:1 assignment (no threshold)")):
        for c, name in zip(SERIES, SETTINGS):
            d = agg[agg["setting"] == name]
            ax.plot(d["train_pairs"], d[metric], color=c, lw=2, marker="o", ms=4, label=name)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        style(ax, title, "validated training pairs (log scale)", metric)
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    axes[1].text(5.5, 0.93, "full and date+amount overlap at 1.0", fontsize=7, color=INK2)
    fig.suptitle("Provided data: ML stage alone, matching all held-out transactions", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "e2_learning_curve.png")
    plt.close(fig)
    return agg


# ---------------------------------------------------------------- E3
def e3_hardest(B, R, gt, seeds):
    """Which categories does the amount-blind model confuse (where text+date must carry the load)?"""
    gt_list = sorted(gt)
    cat = pd.read_csv(ROOT / "data/check_register.csv").set_index("transaction_id")["category"]
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(100 + seed)
        idx = rng.choice(len(gt_list), 200, replace=False)
        train = [gt_list[i] for i in idx]
        test = gt - set(train)
        _, _, props = ml_only(B, R, train, test, MatcherConfig(use_amount=False, seed=seed))
        pred = {p.bank_id: p.reg_id for p in props}
        for b, r in test:
            rows.append({"category": cat[r], "correct": pred.get(b) == r})
    df = pd.DataFrame(rows).groupby("category")["correct"].agg(["mean", "count"]).rename(
        columns={"mean": "assignment_accuracy", "count": "test_cases"}).sort_values("assignment_accuracy")
    df.to_csv(OUT / "e3_hardest_categories_provided.csv")
    return df


# ---------------------------------------------------------------- E4
def e4_ablation(B, R, gt, seeds):
    gt_list = sorted(gt)
    base = MatcherConfig(use_amount=False)
    variants = {
        "default (k=100, MI top-1 align, char-3grams, overlap)": base,
        "k = 10 components": replace(base, svd=replace(base.svd, k=10)),
        "k = 20 components": replace(base, svd=replace(base.svd, k=20)),
        "k = 40 components": replace(base, svd=replace(base.svd, k=40)),
        "no MI alignment (plain cross-language LSI)": replace(base, svd=replace(base.svd, use_alignment=False)),
        "MI alignment top-3 partners": replace(base, svd=replace(base.svd, top_align=3)),
        "no character n-grams": replace(base, svd=replace(base.svd, features=FeatureConfig(char_ngrams=False, amount_terms=False))),
        "no shared-token overlap feature": replace(base, use_overlap=False),
    }
    rows = []
    for name, cfg in variants.items():
        for n in (40, 200):
            res = []
            for seed in range(seeds):
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(gt_list), n, replace=False)
                train = [gt_list[i] for i in idx]
                s, top1, _ = ml_only(B, R, train, gt - set(train), replace(cfg, seed=seed))
                res.append((s.precision, s.recall, s.f1, top1.precision))
            p, r, f, t = np.mean(res, axis=0)
            rows.append({"variant": name, "train_pairs": n, "precision": p, "recall": r, "f1": f, "top1_accuracy": t})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e4_ablation_amount_blind.csv", index=False)
    return df


# ---------------------------------------------------------------- E5
def e5_synthetic(seeds):
    frames = []
    variants = {
        "learning carried across months": SimConfig(),
        "no carry-over (retrain from scratch monthly)": SimConfig(carry_training=False),
        "carried, 5% reviewer error": SimConfig(reviewer_error=0.05),
    }
    stats = []
    for seed in range(seeds):
        b, r, t = generate(SynthConfig(seed=7 + seed, volume=2.0))
        stats.append({"seed": 7 + seed, "bank": len(b), "register": len(r), "true_pairs": len(t)})
        for name, sim in variants.items():
            df = simulate(b, r, t, sim=sim)
            df["variant"], df["seed"] = name, 7 + seed
            frames.append(df)
    df = pd.concat(frames)
    df = df[df["true_pairs"] > 20]  # drop the 3-transaction spill-over tail month
    df.to_csv(OUT / "e5_synthetic_monthly_raw.csv", index=False)
    agg = df.groupby(["variant", "month_idx"])[["ml_precision", "ml_recall", "ml_f1", "auto_precision", "auto_recall",
                                                  "auto_f1", "unique_precision", "true_pair_cosine_prior_model"]].mean().reset_index()
    agg.to_csv(OUT / "e5_synthetic_monthly.csv", index=False)
    tot = df.groupby("variant").apply(lambda g: pd.Series({
        "ml_precision": np.average(g["ml_precision"], weights=g["ml_proposed"].clip(lower=1)),
        "ml_recall": np.average(g["ml_recall"], weights=g["ml_candidates"].clip(lower=1)),
        "auto_precision": np.average(g["auto_precision"], weights=g["unique_proposed"] + g["ml_proposed"]),
        "auto_recall": np.average(g["auto_recall"], weights=g["true_pairs"]),
        "unique_precision": np.average(g["unique_precision"], weights=g["unique_proposed"].clip(lower=1)),
        "unique_share_of_pairs": (g["unique_proposed"].sum() / g["true_pairs"].sum()),
    }), include_groups=False).reset_index()
    tot["ml_f1"] = 2 * tot.ml_precision * tot.ml_recall / (tot.ml_precision + tot.ml_recall)
    tot["auto_f1"] = 2 * tot.auto_precision * tot.auto_recall / (tot.auto_precision + tot.auto_recall)
    tot.to_csv(OUT / "e5_synthetic_totals.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    for c, name in zip(SERIES, variants):
        d = agg[agg["variant"] == name]
        axes[0].plot(d["month_idx"], d["ml_recall"], color=c, lw=2, marker="o", ms=4, label=name)
        axes[1].plot(d["month_idx"], d["ml_precision"], color=c, lw=2, marker="o", ms=4, label=name)
    for ax, t in ((axes[0], "ML-stage recall (non-unique pairs matched automatically)"), (axes[1], "ML-stage precision")):
        ax.axvline(7, color=INK2, lw=1, ls=":")
        ax.text(7.15, 0.08, "bank format\nchanges", fontsize=7, color=INK2)
        ax.set_ylim(0, 1.02)
        style(ax, t, "month", "")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"Synthetic 12-month run, mean of {seeds} seeds", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "e5_synthetic_monthly.png")
    plt.close(fig)
    return tot, agg, pd.DataFrame(stats)


# ---------------------------------------------------------------- E6
def e6_synthetic_curve(seeds):
    """Realistic learning curve: train on the first N validated pairs (chronological), test on the last month
    after unique-amount matching has taken its share."""
    rows = []
    for seed in range(seeds):
        b, r, t = generate(SynthConfig(seed=7 + seed, volume=2.0))
        B, R = bank_frame(b), register_frame(r)
        last = B["date"].dt.to_period("M").max() - 1  # last full month (the final one is a spill-over tail)
        test_b = B[B["date"].dt.to_period("M") == last]
        test_r = R[R["date"].dt.to_period("M") == last]
        train_b = B[B["date"].dt.to_period("M") < last]
        hist = sorted([(x, y) for x, y in t if x in set(train_b["id"]) and y in set(R["id"]) and y not in set(test_r["id"])],
                      key=lambda p: B.set_index("id").loc[p[0], "date"])
        s0 = ReconciliationSession(test_b, test_r)
        u = s0.run_unique()
        s0.review(u, OracleReviewer(t))
        lb, lr = s0.unmatched()
        test_truth = {(x, y) for x, y in t if x in set(lb["id"]) and y in set(lr["id"])}
        for n in (0, 10, 25, 50, 100, 200, 400, 800, 10**6):
            m = MLMatcher(MatcherConfig(seed=seed)).fit(pairs_from_frames(B, R, hist[:n], "gt"))
            props = m.match(lb, lr, background=(test_b, test_r))
            above = [(p.bank_id, p.reg_id) for p in props if p.accepted_by_threshold and p.llr > 0]
            s = evaluate(above, test_truth)
            rows.append({"seed": seed, "train_pairs": min(n, len(hist)) if n < 10**6 else "all (~1100)", "precision": s.precision, "recall": s.recall, "f1": s.f1,
                         "test_pairs": len(test_truth)})
    df = pd.DataFrame(rows).groupby("train_pairs", sort=False)[["precision", "recall", "f1", "test_pairs"]].mean().reset_index()
    df.to_csv(OUT / "e6_synthetic_learning_curve.csv", index=False)
    return df


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f") if hasattr(df, "to_markdown") else df.to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds = 2 if a.quick else 5
    OUT.mkdir(exist_ok=True)
    B, R = load_bank(ROOT / "data/bank_statements.csv"), load_register(ROOT / "data/check_register.csv")
    gt = id_ground_truth(B, R)
    out = ["# Experiment results (generated by scripts/run_experiments.py)\n"]
    t0 = time.time()

    e1, log = e1_provided(B, R, gt)
    out += ["## E1 - provided data, full pipeline\n", "```", *log, "```\n", e1.to_string(index=False), "\n"]
    print("E1 done", round(time.time() - t0))
    e2 = e2_learning_curve(B, R, gt, seeds)
    out += ["## E2 - learning curve (provided data, ML stage only)\n", e2.round(3).to_string(index=False), "\n"]
    print("E2 done", round(time.time() - t0))
    e3 = e3_hardest(B, R, gt, seeds)
    out += ["## E3 - hardest categories (amount-blind, 200 training pairs)\n", e3.round(3).to_string(), "\n"]
    e4 = e4_ablation(B, R, gt, seeds)
    out += ["## E4 - ablations (amount-blind)\n", e4.round(3).to_string(index=False), "\n"]
    print("E3/E4 done", round(time.time() - t0))
    e6 = e6_synthetic_curve(seeds)
    out += ["## E6 - synthetic learning curve (last month, after unique stage)\n", e6.round(3).to_string(index=False), "\n"]
    print("E6 done", round(time.time() - t0))
    tot, agg, stats = e5_synthetic(min(seeds, 3))
    out += ["## E5 - synthetic 12-month simulation\n", stats.to_string(index=False), "\n", tot.round(3).to_string(index=False), "\n",
            agg.round(3).to_string(index=False), "\n"]
    print("E5 done", round(time.time() - t0))
    (OUT / "summary.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
