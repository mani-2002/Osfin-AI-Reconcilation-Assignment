"""Cross-language SVD retrieval model (Chew, ICAIF'20, section 2.1).

Pipeline, for a "parallel corpus" of previously validated (bank, register) pairs:

1. **Term alignment by mutual information.** For every bank term *a* and every
   register term *b* we compute the mutual information of their presence
   indicators across the parallel pairs (paper eq. 1). Each term keeps only
   its best partner(s) - "ruling out all but those pairs that maximize mutual
   information". This yields a sparse, symmetric alignment matrix ``A``
   (``I`` on the diagonal so every term still maps onto itself).
2. **Term-by-transaction matrix.** Each training pair becomes one "document"
   (the concatenation of both sides' terms, as in cross-language LSI),
   weighted with log-entropy. Call it ``D`` (terms x pairs).
3. **Reinforce and factorise.** ``X = A @ D`` then ``X = U S V^T`` (truncated
   to ``k`` components).
4. **Projection.** A new transaction from *either* source becomes a weighted
   term vector ``t``; its embedding is ``U_k^T (A t)``, L2-normalised. Because
   aligned terms co-occur in the training documents and are coupled through
   ``A``, a bank query lands next to its register counterpart even when the two
   share no surface vocabulary.

Departures from the paper (see docs/ANALYSIS.md for discussion):

* MI is *normalised* by ``min(H(a), H(b))`` before entering ``A`` so the
  alignment weight sits in [0, 1] and is comparable to the identity diagonal.
* Only positively associated term pairs are aligned (MI is symmetric in sign,
  so a strongly *anti*-correlated pair would otherwise be aligned).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from scipy import sparse

from .features import FeatureConfig


def _svd(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thin SVD (U, S). LAPACK's divide-and-conquer driver (gesdd, numpy's default) occasionally
    fails to converge on well-conditioned input; fall back to the slower but robust gesvd."""
    try:
        U, S, _ = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        U, S, _ = scipy.linalg.svd(X, full_matrices=False, lapack_driver="gesvd")
    return U, S


def _xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    mask = x > 0
    out[mask] = x[mask] * np.log(y[mask])
    return out


def binary_mutual_information(n11: np.ndarray, n1_: np.ndarray, n_1: np.ndarray, n: int) -> np.ndarray:
    """Mutual information (nats) between binary presence variables.

    ``n11[i, j]`` counts documents containing both term i and term j; ``n1_[i]``
    and ``n_1[j]`` are the marginal document frequencies; ``n`` the number of
    documents. Implements eq. (1) of the paper summed over the four cells of
    the 2x2 contingency table.
    """
    n11 = n11.astype(float)
    a = n1_[:, None].astype(float)
    b = n_1[None, :].astype(float)
    n10 = a - n11
    n01 = b - n11
    n00 = n - n11 - n10 - n01
    mi = np.zeros_like(n11)
    for cell, pa, pb in (
        (n11, a, b),
        (n10, a, n - b),
        (n01, n - a, b),
        (n00, n - a, n - b),
    ):
        p_joint = cell / n
        denom = (pa / n) * (pb / n)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denom > 0, p_joint / np.where(denom > 0, denom, 1), 1.0)
        mi += _xlogy(p_joint, ratio)
    return np.maximum(mi, 0.0)


def binary_entropy(df: np.ndarray, n: int) -> np.ndarray:
    p = np.clip(df / n, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


@dataclass
class SVDConfig:
    k: int = 100                 # principal components (paper tested 20-40; more was better)
    top_align: int = 1           # partners kept per term during alignment
    use_alignment: bool = True   # ablation switch: plain cross-language LSI when False
    min_df: int = 1              # ignore terms seen in fewer training pairs
    features: FeatureConfig = FeatureConfig()


class AlignedSVDModel:
    """Learns a shared semantic space for bank and register transactions."""

    def __init__(self, config: SVDConfig | None = None):
        self.cfg = config or SVDConfig()
        self.vocab: dict[str, int] = {}
        self.global_w: np.ndarray | None = None
        self.A: sparse.csr_matrix | None = None
        self.U: np.ndarray | None = None
        self.S: np.ndarray | None = None
        self.alignments: list[tuple[str, str, float]] = []

    # ------------------------------------------------------------------ fit
    def fit(self, bank_terms: list[list[str]], reg_terms: list[list[str]]) -> "AlignedSVDModel":
        if len(bank_terms) != len(reg_terms):
            raise ValueError("bank_terms and reg_terms must be parallel")
        n = len(bank_terms)
        if n == 0:
            raise ValueError("need at least one training pair")
        docs = [b + r for b, r in zip(bank_terms, reg_terms)]

        df_counts: dict[str, int] = {}
        for d in docs:
            for t in set(d):
                df_counts[t] = df_counts.get(t, 0) + 1
        terms = sorted(t for t, c in df_counts.items() if c >= self.cfg.min_df)
        self.vocab = {t: i for i, t in enumerate(terms)}
        V = len(terms)

        tf = self._count_matrix(docs)                      # n x V (raw counts)
        self.global_w = self._entropy_weights(tf, n)
        D = (tf.log1p().multiply(self.global_w[None, :])).T.tocsr()   # V x n

        self.A = self._alignment_matrix(tf, n, V) if self.cfg.use_alignment else sparse.identity(V, format="csr")
        X = (self.A @ D).toarray()

        U, S = _svd(X)
        k = max(1, min(self.cfg.k, int((S > 1e-10).sum())))
        self.U, self.S = U[:, :k], S[:k]
        return self

    def _count_matrix(self, docs: list[list[str]]) -> sparse.csr_matrix:
        rows, cols, vals = [], [], []
        for i, d in enumerate(docs):
            counts: dict[int, int] = {}
            for t in d:
                j = self.vocab.get(t)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            rows += [i] * len(counts)
            cols += list(counts.keys())
            vals += list(counts.values())
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)), dtype=float)

    @staticmethod
    def _entropy_weights(tf: sparse.csr_matrix, n: int) -> np.ndarray:
        """Global log-entropy weight: 1 + sum_d p log p / log n (1 = informative, 0 = everywhere)."""
        if n < 2:
            return np.ones(tf.shape[1])
        gf = np.asarray(tf.sum(axis=0)).ravel()
        coo = tf.tocoo()
        p = coo.data / gf[coo.col]
        plogp = np.zeros(tf.shape[1])
        np.add.at(plogp, coo.col, p * np.log(p))
        return 1.0 + plogp / np.log(n)

    def _alignment_matrix(self, tf: sparse.csr_matrix, n: int, V: int) -> sparse.csr_matrix:
        terms = np.array(list(self.vocab.keys()))
        is_b = np.char.startswith(terms.astype(str), "b:")
        b_idx, r_idx = np.where(is_b)[0], np.where(~is_b)[0]
        P = (tf > 0).astype(float).tocsc()
        Pb, Pr = P[:, b_idx], P[:, r_idx]
        n11 = (Pb.T @ Pr).toarray()
        dfb = np.asarray(Pb.sum(axis=0)).ravel()
        dfr = np.asarray(Pr.sum(axis=0)).ravel()

        mi = binary_mutual_information(n11, dfb, dfr, n)
        positive = n11 * n > dfb[:, None] * dfr[None, :]
        hmin = np.minimum(binary_entropy(dfb, n)[:, None], binary_entropy(dfr, n)[None, :])
        nmi = np.where(positive & (hmin > 0), mi / np.maximum(hmin, 1e-12), 0.0)
        nmi = np.clip(nmi, 0.0, 1.0)

        keep = np.zeros_like(nmi, dtype=bool)
        k = self.cfg.top_align
        if nmi.size:
            # best partner(s) for every bank term (rows) and every register term (cols)
            for axis in (1, 0):
                order = np.argsort(-nmi, axis=axis)
                top = np.take(order, range(min(k, nmi.shape[axis])), axis=axis)
                if axis == 1:
                    rr = np.repeat(np.arange(nmi.shape[0]), top.shape[1])
                    keep[rr, top.ravel()] = True
                else:
                    cc = np.tile(np.arange(nmi.shape[1]), top.shape[0])
                    keep[top.ravel(), cc] = True
        keep &= nmi > 0

        bi, rj = np.nonzero(keep)
        w = nmi[bi, rj]
        gb, gr = b_idx[bi], r_idx[rj]
        self.alignments = sorted(
            ((terms[a], terms[b], float(x)) for a, b, x in zip(gb, gr, w)), key=lambda t: -t[2]
        )
        rows = np.concatenate([np.arange(V), gb, gr])
        cols = np.concatenate([np.arange(V), gr, gb])
        vals = np.concatenate([np.ones(V), w, w])
        return sparse.csr_matrix((vals, (rows, cols)), shape=(V, V))

    # ------------------------------------------------------------- project
    def embed(self, term_lists: list[list[str]]) -> np.ndarray:
        """Project transactions into the shared space; rows are unit vectors (zero if no known terms)."""
        if self.U is None:
            raise RuntimeError("model is not fitted")
        tf = self._count_matrix(term_lists)
        T = tf.log1p().multiply(self.global_w[None, :]).tocsr()
        Z = np.asarray((T @ self.A) @ self.U)
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        return np.divide(Z, norms, out=np.zeros_like(Z), where=norms > 0)

    def similarity(self, bank_terms: list[list[str]], reg_terms: list[list[str]]) -> np.ndarray:
        """Cosine similarity matrix (bank x register) in the reduced space."""
        return self.embed(bank_terms) @ self.embed(reg_terms).T

    def top_alignments(self, n: int = 20, family: str | None = "w") -> list[tuple[str, str, float]]:
        """Most strongly aligned (bank term, register term) pairs - useful for explaining the model."""
        out = [a for a in self.alignments if family is None or (a[0].split(":")[1] == family and a[1].split(":")[1] == family)]
        return out[:n]
