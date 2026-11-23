"""Stage 1: unique-amount matching (paper section 2.2, first automated step).

If an amount occurs exactly once in the bank statement and exactly once in the
check register, the two are paired. This needs no training data and is what
*seeds* the parallel corpus for the ML stage.

Confidence is a statistical, not ad-hoc, quantity: we estimate how likely it is
that an unrelated register entry would carry the same amount *by chance*
(using a kernel density estimate of the register's amount distribution) and
discount it by how plausible the date lag and type are.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Flag codes surfaced to the reviewer
LAG_EXCEEDS = "DATE_LAG_EXCEEDS_THRESHOLD"
REGISTER_AFTER_BANK = "REGISTER_DATED_AFTER_BANK"
TYPE_MISMATCH = "TYPE_MISMATCH"
SOURCE_NOTE = "SOURCE_HAS_ANOMALY_NOTE"
LOW_TEXT_SIM = "DESCRIPTIONS_DISAGREE"
AMOUNT_DIFF = "AMOUNT_DIFFERS"
LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass
class UniqueConfig:
    max_lag_days: int = 5        # expected register->bank lag window (from the brief)
    min_lag_days: int = 0
    tolerance: float = 0.0       # amount tolerance; 0 = exact match only
    lag_decay_days: float = 3.0  # how fast confidence decays outside the window
    type_mismatch_penalty: float = 0.5


@dataclass
class UniqueMatch:
    bank_id: str
    reg_id: str
    amount: float
    lag: int
    confidence: float
    chance_prob: float
    flags: list[str] = field(default_factory=list)


def _chance_probability(amount: float, all_amounts: np.ndarray, tol: float) -> float:
    """P(at least one *unrelated* entry lands within +/- (tol or half a cent) of ``amount``).

    Uses a Gaussian KDE on log-amounts (Silverman bandwidth) to get the local
    density, then a Poisson approximation over the other n-1 entries.
    """
    x = np.log(np.maximum(all_amounts, 0.01))
    n = len(x)
    if n < 2:
        return 0.0
    h = 1.06 * np.std(x) * n ** (-1 / 5) or 0.1
    lx = np.log(max(amount, 0.01))
    dens_log = np.exp(-0.5 * ((lx - x) / h) ** 2).sum() / (n * h * np.sqrt(2 * np.pi))
    dens = dens_log / max(amount, 0.01)                   # change of variables to $ scale
    width = 2 * max(tol, 0.005)
    expected = (n - 1) * dens * width
    return float(1 - np.exp(-expected))


def unique_amount_matches(bank: pd.DataFrame, reg: pd.DataFrame, cfg: UniqueConfig | None = None) -> list[UniqueMatch]:
    """Pair transactions whose amount is unique in *both* sources."""
    cfg = cfg or UniqueConfig()
    if len(bank) == 0 or len(reg) == 0:
        return []
    b_amt = bank["amount"].round(2).values
    r_amt = reg["amount"].round(2).values
    # number of entries within tolerance, per side
    b_cnt_b = (np.abs(b_amt[:, None] - b_amt[None, :]) <= cfg.tolerance + 1e-9).sum(1)
    r_cnt_r = (np.abs(r_amt[:, None] - r_amt[None, :]) <= cfg.tolerance + 1e-9).sum(1)
    close = np.abs(b_amt[:, None] - r_amt[None, :]) <= cfg.tolerance + 1e-9   # bank x reg

    out = []
    all_reg = r_amt
    for i in np.where(b_cnt_b == 1)[0]:
        js = np.where(close[i])[0]
        if len(js) != 1:
            continue
        j = js[0]
        if r_cnt_r[j] != 1 or close[:, j].sum() != 1:
            continue
        b, r = bank.iloc[i], reg.iloc[j]
        lag = int((b["date"] - r["date"]).days)
        flags = []
        conf = 1.0 - _chance_probability(float(b["amount"]), all_reg, cfg.tolerance)
        chance = 1.0 - conf
        if lag > cfg.max_lag_days or lag < cfg.min_lag_days:
            flags.append(LAG_EXCEEDS)
            dist = lag - cfg.max_lag_days if lag > cfg.max_lag_days else cfg.min_lag_days - lag
            conf *= float(np.exp(-dist / cfg.lag_decay_days))
        if lag < 0:
            flags.append(REGISTER_AFTER_BANK)
        if b["direction"] and r["direction"] and b["direction"] != r["direction"]:
            flags.append(TYPE_MISMATCH)
            conf *= cfg.type_mismatch_penalty
        if r["notes"] or b["notes"]:
            flags.append(SOURCE_NOTE)
        if abs(b["amount"] - r["amount"]) > 1e-9:
            flags.append(AMOUNT_DIFF)
        out.append(UniqueMatch(b["id"], r["id"], float(b["amount"]), lag, round(conf, 4), round(chance, 6), flags))
    return out
