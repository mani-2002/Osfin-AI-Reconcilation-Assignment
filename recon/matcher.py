"""ML matching: SVD text similarity + learned date/amount evidence -> 1:1 assignment.

The SVD model (:mod:`recon.svd_model`) tells us how *semantically* close two
descriptions are. On its own that only identifies the *kind* of transaction
(e.g. "a grocery purchase"), so we combine it with the other evidence a human
would use - date lag and amount discrepancy - in a Fellegi-Sunter style
likelihood-ratio model whose every parameter is *learned*:

    log LR(i, j) = log P(cos | match)/P(cos | non-match)
                 + log P(lag | match)/P(lag | non-match)
                 + log P(d_amount | match)/P(d_amount | non-match)

plus a learned term for literally shared tokens (cheque numbers etc.).

* ``P(. | match)`` comes from the validated training pairs (for the cosine we
  use **cross-fitted** scores so the model is not graded on pairs it was
  trained on).
* ``P(. | non-match)`` comes from the batch being reconciled itself: almost
  all of the N x M candidate pairs are non-matches, so their empirical
  distribution is a good unsupervised estimate.

So nothing about "dates are usually 0-5 days apart" or "amounts may differ by
a few cents" is hard-coded: the system learns it from what the reviewer has
accepted, and adapts if the bank's behaviour changes.

Confidence is a posterior: for bank row i, ``P(j | i) = LR_ij / (sum_j' LR_ij'
+ N (1-rho)/rho)`` where ``rho`` is the prior probability that a transaction
has a counterpart at all. We compute it both row- and column-wise and take
the minimum, so two indistinguishable candidates (same amount, same day, same
description) each get ~50% and are routed to a human instead of guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .features import FeatureConfig, frame_terms
from .svd_model import AlignedSVDModel, SVDConfig

LAG_EDGES = np.arange(-15.5, 31.5, 1.0)                     # 1-day bins, clipped at the ends
AMT_EDGES = np.array([-0.5, 0.5, 1.5, 2.5, 5.5, 10.5, 25.5, 100.5, 1000.5, 1e12])  # cents
COS_EDGES = np.linspace(-0.3, 1.0001, 27)
OVL_EDGES = np.array([-0.5, 0.5, 1.5, 2.5, 100.0])          # shared raw tokens: 0, 1, 2, 3+
_RAW_TOKEN = re.compile(r"[a-z0-9]{2,}")


def raw_tokens(text: str) -> frozenset[str]:
    """Raw tokens *including* numbers - lets identifiers such as cheque numbers count."""
    return frozenset(_RAW_TOKEN.findall(text.lower()))


def token_overlap(bank_texts, reg_texts) -> np.ndarray:
    """Number of raw tokens shared by each (bank, register) pair, via a sparse inverted index."""
    bt = [raw_tokens(t) for t in bank_texts]
    rt = [raw_tokens(t) for t in reg_texts]
    out = np.zeros((len(bt), len(rt)))
    index: dict[str, list[int]] = {}
    for j, toks in enumerate(rt):
        for t in toks:
            index.setdefault(t, []).append(j)
    for i, toks in enumerate(bt):
        for t in toks:
            for j in index.get(t, ()):
                out[i, j] += 1
    return out


class SmoothedHistogram:
    """1-D histogram density with additive smoothing (optionally decaying pseudo-counts)."""

    def __init__(self, edges: np.ndarray, alpha: float = 0.5, decay: float | None = None):
        self.edges = edges
        self.alpha = alpha
        self.decay = decay
        self.p: np.ndarray | None = None

    def _bin(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=float), self.edges[0] + 1e-9, self.edges[-1] - 1e-9)
        return np.searchsorted(self.edges, x, side="right") - 1

    def fit(self, x: np.ndarray, weights: np.ndarray | None = None) -> "SmoothedHistogram":
        nb = len(self.edges) - 1
        counts = np.bincount(self._bin(np.ravel(x)), weights=None if weights is None else np.ravel(weights), minlength=nb)
        prior = np.full(nb, self.alpha)
        if self.decay is not None:
            prior = self.alpha * self.decay ** np.arange(nb)
        c = counts[:nb] + prior
        self.p = c / c.sum()
        return self

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self.p[self._bin(x)]


@dataclass
class MatcherConfig:
    svd: SVDConfig = field(default_factory=SVDConfig)
    use_text: bool = True
    use_date: bool = True
    use_amount: bool = True
    use_overlap: bool = True     # learned weight for literally shared tokens (e.g. cheque numbers)
    rho: float = 0.95            # prior P(a transaction has a counterpart)
    llr_clip: float = 8.0        # cap per-feature evidence to avoid over-confidence
    min_pairs_for_text: int = 8  # below this, the SVD model is too thin to trust
    cv_folds: int = 5
    accept_threshold: float = 0.6
    seed: int = 0


@dataclass
class TrainingPair:
    """A validated (bank, register) match - one sentence pair of the parallel corpus."""
    bank_text: str
    bank_type: str
    bank_amount: float
    bank_date: pd.Timestamp
    reg_text: str
    reg_type: str
    reg_amount: float
    reg_date: pd.Timestamp
    source: str = "unknown"      # unique | ml | manual

    @property
    def lag(self) -> int:
        return int((self.bank_date - self.reg_date).days)

    @property
    def amount_diff_cents(self) -> float:
        return abs(round((self.bank_amount - self.reg_amount) * 100))


def pairs_from_frames(bank: pd.DataFrame, reg: pd.DataFrame, pairs: list[tuple[str, str]], source: str) -> list[TrainingPair]:
    b = bank.set_index("id")
    r = reg.set_index("id")
    out = []
    for bid, rid in pairs:
        x, y = b.loc[bid], r.loc[rid]
        out.append(TrainingPair(x.text, x.type, float(x.amount), x.date, y.text, y.type, float(y.amount), y.date, source))
    return out


@dataclass
class Proposal:
    bank_id: str
    reg_id: str
    confidence: float
    llr: float
    cosine: float
    lag: int
    amount_diff: float
    accepted_by_threshold: bool


class MLMatcher:
    """Train on validated pairs, then propose 1:1 matches for unreconciled transactions."""

    def __init__(self, config: MatcherConfig | None = None):
        self.cfg = config or MatcherConfig()
        fc = self.cfg.svd.features
        if not self.cfg.use_amount and fc.amount_terms:
            # keep the "amount-blind" ablation honest: no amount info anywhere
            self.cfg.svd = SVDConfig(**{**self.cfg.svd.__dict__, "features": FeatureConfig(**{**fc.__dict__, "amount_terms": False})})
        self.model: AlignedSVDModel | None = None
        self.lag_m = SmoothedHistogram(LAG_EDGES, alpha=0.2)
        self.amt_m = SmoothedHistogram(AMT_EDGES, alpha=0.5, decay=0.5)
        self.cos_m = SmoothedHistogram(COS_EDGES, alpha=0.5)
        self.ovl_m = SmoothedHistogram(OVL_EDGES, alpha=0.5)
        self.n_train = 0

    # ---------------------------------------------------------------- utils
    def _terms(self, texts, types, amounts, side):
        df = pd.DataFrame({"text": texts, "type": types, "amount": amounts})
        return frame_terms(df, side, self.cfg.svd.features)

    def _pair_terms(self, pairs: list[TrainingPair]):
        bt = self._terms([p.bank_text for p in pairs], [p.bank_type for p in pairs], [p.bank_amount for p in pairs], "b")
        rt = self._terms([p.reg_text for p in pairs], [p.reg_type for p in pairs], [p.reg_amount for p in pairs], "r")
        return bt, rt

    @property
    def text_enabled(self) -> bool:
        return self.cfg.use_text and self.n_train >= self.cfg.min_pairs_for_text

    # ------------------------------------------------------------------ fit
    def fit(self, pairs: list[TrainingPair]) -> "MLMatcher":
        self.n_train = len(pairs)
        if self.n_train == 0:
            self.model = None
            self.lag_m.fit(np.array([]))
            self.amt_m.fit(np.array([]))
            self.cos_m.fit(np.array([]))
            self.ovl_m.fit(np.array([]))
            return self
        self.lag_m.fit(np.array([p.lag for p in pairs]))
        self.amt_m.fit(np.array([p.amount_diff_cents for p in pairs]))
        self.ovl_m.fit(np.array([len(raw_tokens(p.bank_text) & raw_tokens(p.reg_text)) for p in pairs]))
        if self.text_enabled:
            bt, rt = self._pair_terms(pairs)
            self.model = AlignedSVDModel(self.cfg.svd).fit(bt, rt)
            self.cos_m.fit(self._crossfit_cosines(bt, rt))
        else:
            self.model = None
        return self

    def _crossfit_cosines(self, bt, rt) -> np.ndarray:
        """Cosine of each training pair under a model that did NOT see that pair."""
        n = len(bt)
        folds = min(self.cfg.cv_folds, n)
        idx = np.random.default_rng(self.cfg.seed).permutation(n)
        out = np.empty(n)
        for f in range(folds):
            test = idx[f::folds]
            train = np.setdiff1d(idx, test)
            if len(train) == 0:
                out[test] = 0.0
                continue
            m = AlignedSVDModel(self.cfg.svd).fit([bt[i] for i in train], [rt[i] for i in train])
            eb, er = m.embed([bt[i] for i in test]), m.embed([rt[i] for i in test])
            out[test] = (eb * er).sum(axis=1)
        return out

    # ---------------------------------------------------------------- score
    def _raw_features(self, bank: pd.DataFrame, reg: pd.DataFrame) -> dict[str, np.ndarray]:
        lag = (bank["date"].values[:, None] - reg["date"].values[None, :]) / np.timedelta64(1, "D")
        damt = np.abs(np.round((bank["amount"].values[:, None] - reg["amount"].values[None, :]) * 100))
        cos = np.zeros(lag.shape)
        if self.model is not None:
            bt = self._terms(bank["text"], bank["type"], bank["amount"], "b")
            rt = self._terms(reg["text"], reg["type"], reg["amount"], "r")
            cos = self.model.similarity(bt, rt)
        ovl = token_overlap(bank["text"], reg["text"]) if self.cfg.use_overlap else np.zeros(lag.shape)
        return {"lag": lag, "damt": damt, "cos": cos, "ovl": ovl}

    def score(self, bank: pd.DataFrame, reg: pd.DataFrame,
              background: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> dict[str, np.ndarray]:
        """Evidence matrices (bank rows x register cols) for every candidate pair.

        ``background`` (full bank, full register) is used to estimate the
        non-match distributions; by default the batch itself is used. Passing
        the full period is more stable when only a handful of transactions are
        left to match.
        """
        f = self._raw_features(bank, reg)
        g = f if background is None else self._raw_features(*background)
        clip = self.cfg.llr_clip

        def llr(m_hist: SmoothedHistogram, key: str, edges: np.ndarray) -> np.ndarray:
            u_hist = SmoothedHistogram(edges, alpha=0.5).fit(g[key])
            return np.clip(np.log(m_hist.pdf(f[key])) - np.log(u_hist.pdf(f[key])), -clip, clip)

        out = {"lag": f["lag"], "amount_diff": f["damt"] / 100.0, "cosine": f["cos"]}
        total = np.zeros(f["lag"].shape)
        if self.n_train > 0 and self.cfg.use_date:
            out["llr_date"] = llr(self.lag_m, "lag", LAG_EDGES)
            total += out["llr_date"]
        if self.n_train > 0 and self.cfg.use_amount:
            out["llr_amount"] = llr(self.amt_m, "damt", AMT_EDGES)
            total += out["llr_amount"]
        if self.model is not None:
            out["llr_text"] = llr(self.cos_m, "cos", COS_EDGES)
            total += out["llr_text"]
        if self.n_train > 0 and self.cfg.use_overlap:
            out["llr_overlap"] = llr(self.ovl_m, "ovl", OVL_EDGES)
            total += out["llr_overlap"]
        out["llr"] = total
        out["confidence"] = self._posterior(total)
        return out

    def _posterior(self, llr: np.ndarray) -> np.ndarray:
        """min(row-wise, column-wise) posterior that (i, j) is the true pair."""
        rho = self.cfg.rho
        L = np.exp(llr - llr.max()) if llr.size else llr
        scale = np.exp(-llr.max()) if llr.size else 1.0
        nr, nc = llr.shape
        null_r = nc * (1 - rho) / rho * scale
        null_c = nr * (1 - rho) / rho * scale
        p_row = L / (L.sum(axis=1, keepdims=True) + null_r)
        p_col = L / (L.sum(axis=0, keepdims=True) + null_c)
        return np.minimum(p_row, p_col)

    # ---------------------------------------------------------------- match
    def match(self, bank: pd.DataFrame, reg: pd.DataFrame,
              background: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> list[Proposal]:
        """Globally optimal 1:1 assignment, each proposal tagged with its confidence."""
        if len(bank) == 0 or len(reg) == 0:
            return []
        ev = self.score(bank, reg, background)
        rows, cols = linear_sum_assignment(-ev["llr"])
        props = []
        for i, j in zip(rows, cols):
            conf = float(ev["confidence"][i, j])
            props.append(Proposal(
                bank_id=bank["id"].iloc[i], reg_id=reg["id"].iloc[j], confidence=conf,
                llr=float(ev["llr"][i, j]), cosine=float(ev["cosine"][i, j]),
                lag=int(ev["lag"][i, j]), amount_diff=float(ev["amount_diff"][i, j]),
                accepted_by_threshold=conf >= self.cfg.accept_threshold,
            ))
        return sorted(props, key=lambda p: -p.confidence)
