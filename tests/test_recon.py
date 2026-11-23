"""Unit tests for the critical pieces. Run with ``pytest -q``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recon import (AlignedSVDModel, MatcherConfig, MLMatcher, OracleReviewer, ReconciliationSession,
                   ThresholdReviewer, TrainingStore, evaluate, id_ground_truth, load_bank, load_register,
                   oracle_resolver, pairs_from_frames, unique_amount_matches)
from recon.data import bank_frame, register_frame
from recon.features import char_ngrams, tokenize
from recon.matcher import SmoothedHistogram, token_overlap
from recon.svd_model import binary_mutual_information
from recon.synth import SynthConfig, generate
from recon.unique import LAG_EXCEEDS, TYPE_MISMATCH

DATA = Path(__file__).resolve().parents[1] / "data"


def make(bank_rows, reg_rows):
    b = pd.DataFrame(bank_rows, columns=["transaction_id", "date", "description", "amount", "type"])
    r = pd.DataFrame(reg_rows, columns=["transaction_id", "date", "description", "amount", "type", "category", "notes"])
    return bank_frame(b), register_frame(r)


# ------------------------------------------------------------------- metrics
def test_metrics_definitions():
    s = evaluate([("a", "1"), ("b", "2"), ("c", "9")], [("a", "1"), ("b", "2"), ("d", "4"), ("e", "5")])
    assert s.tp == 2 and s.precision == pytest.approx(2 / 3) and s.recall == pytest.approx(0.5)
    assert s.f1 == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))


def test_metrics_empty():
    assert evaluate([], [("a", "1")]).f1 == 0.0


# ------------------------------------------------------------------ features
def test_tokenize_and_ngrams():
    assert tokenize("BP GAS #1775") == ["bp", "gas", "<num>"]
    assert char_ngrams("gas") == ["^ga", "gas", "as$"]
    assert char_ngrams("<num>") == []


# -------------------------------------------------------------------- unique
def test_unique_matching_basic_and_flags():
    b, r = make(
        [("B1", "2023-01-10", "KROGER", 10.00, "DEBIT"),
         ("B2", "2023-01-10", "ATM", 20.00, "DEBIT"),
         ("B3", "2023-01-11", "ATM", 20.00, "DEBIT"),
         ("B4", "2023-01-30", "PAYROLL", 999.99, "CREDIT")],
        [("R1", "2023-01-08", "Groceries", 10.00, "DR", "Grocery", ""),
         ("R2", "2023-01-09", "Cash", 20.00, "DR", "ATM", ""),
         ("R3", "2023-01-10", "Cash", 20.00, "DR", "ATM", ""),
         ("R4", "2023-01-02", "Salary", 999.99, "DR", "Payroll", "")],
    )
    m = {x.bank_id: x for x in unique_amount_matches(b, r)}
    assert set(m) == {"B1", "B4"}                       # 20.00 is not unique
    assert m["B1"].reg_id == "R1" and not m["B1"].flags
    assert LAG_EXCEEDS in m["B4"].flags and TYPE_MISMATCH in m["B4"].flags
    assert m["B4"].confidence < m["B1"].confidence


def test_unique_requires_uniqueness_on_both_sides():
    b, r = make([("B1", "2023-01-10", "X", 5.0, "DEBIT")],
                [("R1", "2023-01-10", "x", 5.0, "DR", "", ""), ("R2", "2023-01-11", "y", 5.0, "DR", "", "")])
    assert unique_amount_matches(b, r) == []


def test_unique_on_provided_data_is_perfectly_precise():
    b, r = load_bank(DATA / "bank_statements.csv"), load_register(DATA / "check_register.csv")
    gt = id_ground_truth(b, r)
    found = unique_amount_matches(b, r)
    s = evaluate([(x.bank_id, x.reg_id) for x in found], gt)
    assert s.precision == 1.0 and s.tp > 250


# ------------------------------------------------------------------ SVD model
def test_mutual_information_known_values():
    # perfectly co-occurring binary variables with p=0.5 -> MI = ln 2
    mi = binary_mutual_information(np.array([[5]]), np.array([5]), np.array([5]), 10)
    assert mi[0, 0] == pytest.approx(np.log(2))
    # independent -> 0
    mi = binary_mutual_information(np.array([[25]]), np.array([50]), np.array([50]), 100)
    assert mi[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_svd_learns_cross_language_alignment():
    """Bank 'kroger' and register 'groceries' share no characters; the model must learn they align."""
    bank = [["b:w:kroger"], ["b:w:shell"], ["b:w:netflix"]] * 6
    reg = [["r:w:groceries"], ["r:w:fuel"], ["r:w:streaming"]] * 6
    m = AlignedSVDModel().fit(bank, reg)
    sim = m.similarity([["b:w:kroger"], ["b:w:shell"]], [["r:w:groceries"], ["r:w:fuel"], ["r:w:streaming"]])
    assert sim[0].argmax() == 0 and sim[1].argmax() == 1
    assert ("b:w:kroger", "r:w:groceries") in {(a, b) for a, b, _ in m.alignments}


def test_svd_unknown_terms_embed_to_zero():
    m = AlignedSVDModel().fit([["b:w:a"], ["b:w:b"]], [["r:w:x"], ["r:w:y"]])
    assert np.allclose(m.embed([["b:w:never_seen"]]), 0)


# ------------------------------------------------------------------- matcher
def test_smoothed_histogram_is_a_distribution():
    h = SmoothedHistogram(np.array([-0.5, 0.5, 1.5, 2.5]), alpha=0.5).fit(np.array([0, 0, 1]))
    assert h.p.sum() == pytest.approx(1.0)
    assert h.pdf(np.array([0]))[0] > h.pdf(np.array([2]))[0]
    assert h.pdf(np.array([99]))[0] == h.pdf(np.array([2]))[0]  # clipped into the end bin


def test_token_overlap_counts_shared_identifiers():
    o = token_overlap(["CHECK #3975", "KROGER"], ["Check #3975", "Groceries"])
    assert o[0, 0] == 2 and o[1, 1] == 0


def test_ml_matcher_resolves_duplicate_amounts_with_text_and_date():
    """Two 100.00 transactions on the same day: only the description can tell them apart."""
    train_b, train_r = [], []
    for i in range(12):
        d = f"2023-01-{i + 1:02d}"
        train_b += [(f"TB{i}a", d, "ATM WITHDRAWAL", 40.0 + i, "DEBIT"), (f"TB{i}b", d, "XFER TO SAVINGS", 300.0 + i, "DEBIT")]
        train_r += [(f"TR{i}a", d, "Cash", 40.0 + i, "DR", "Atm", ""), (f"TR{i}b", d, "Move to savings", 300.0 + i, "DR", "Transfer", "")]
    tb, tr = make(train_b, train_r)
    pairs = [(f"TB{i}{s}", f"TR{i}{s}") for i in range(12) for s in "ab"]
    m = MLMatcher(MatcherConfig(min_pairs_for_text=4)).fit(pairs_from_frames(tb, tr, pairs, "gt"))
    b, r = make([("B1", "2023-02-01", "ATM WITHDRAWAL", 100.0, "DEBIT"), ("B2", "2023-02-01", "XFER TO SAVINGS", 100.0, "DEBIT")],
                [("R1", "2023-02-01", "Move to savings", 100.0, "DR", "Transfer", ""), ("R2", "2023-02-01", "Cash", 100.0, "DR", "Atm", "")])
    props = {p.bank_id: p for p in m.match(b, r, background=(pd.concat([tb, b]), pd.concat([tr, r])))}
    assert props["B1"].reg_id == "R2" and props["B2"].reg_id == "R1"
    assert all(p.confidence > 0.5 for p in props.values())


def test_indistinguishable_duplicates_get_low_confidence():
    b, r = make([("B1", "2023-01-05", "ATM", 100.0, "DEBIT"), ("B2", "2023-01-05", "ATM", 100.0, "DEBIT")],
                [("R1", "2023-01-04", "Cash", 100.0, "DR", "", ""), ("R2", "2023-01-04", "Cash", 100.0, "DR", "", "")])
    tb, tr = make([("T1", "2023-01-01", "ATM", 60.0, "DEBIT")], [("U1", "2022-12-31", "Cash", 60.0, "DR", "", "")])
    m = MLMatcher().fit(pairs_from_frames(tb, tr, [("T1", "U1")], "gt"))
    assert all(p.confidence <= 0.5 for p in m.match(b, r))


# ------------------------------------------------------------------ workflow
def test_full_cycle_on_provided_data():
    b, r = load_bank(DATA / "bank_statements.csv"), load_register(DATA / "check_register.csv")
    gt = id_ground_truth(b, r)
    s = ReconciliationSession(b, r)
    s.run_cycle(OracleReviewer(gt), oracle_resolver(gt))
    sc = s.scores(gt)
    assert sc["automatic"].precision >= 0.99 and sc["automatic"].recall >= 0.95
    assert sc["final_after_qc"].recall == 1.0
    assert len(s.store) == len(gt)


def test_rejected_matches_are_not_reproposed_and_not_learned():
    b, r = load_bank(DATA / "bank_statements.csv"), load_register(DATA / "check_register.csv")
    s = ReconciliationSession(b, r)
    u = s.run_unique()
    s.review(u, ThresholdReviewer(2.0))  # reject everything
    assert len(s.store) == 0
    props, sugg = s.run_ml()
    rejected = {(m.bank_id, m.reg_id) for m in u}
    assert not rejected & {(p.bank_id, p.reg_id) for p in props + sugg}


def test_training_store_roundtrip(tmp_path):
    b, r = load_bank(DATA / "bank_statements.csv"), load_register(DATA / "check_register.csv")
    pairs = pairs_from_frames(b, r, [("B0047", "R0047"), ("B0092", "R0092")], "unique")
    st = TrainingStore(pairs)
    st.save(tmp_path / "s.json")
    back = TrainingStore.load(tmp_path / "s.json")
    assert len(back) == 2 and back.pairs[0].lag == pairs[0].lag and back.pairs[0].bank_text == "BP GAS #1775"


def test_more_training_data_does_not_hurt_on_synthetic():
    bank, reg, truth = generate(SynthConfig(months=3, seed=1, volume=1.5, drift_month=None))
    B, R = bank_frame(bank), register_frame(reg)
    gt = sorted(truth)
    rng = np.random.default_rng(0)
    rng.shuffle(gt)
    test = set(gt[-60:])
    tb, tr = {x for x, _ in test}, {y for _, y in test}

    def f1(n):
        m = MLMatcher(MatcherConfig(use_amount=False)).fit(pairs_from_frames(B, R, gt[:n], "gt"))
        props = m.match(B[B["id"].isin(tb)], R[R["id"].isin(tr)])
        return evaluate([(p.bank_id, p.reg_id) for p in props], test).precision

    assert f1(len(gt) - 60) >= f1(10)
