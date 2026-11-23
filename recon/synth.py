"""Synthetic multi-month data generator (extension of the provided dataset).

The provided 308-transaction file is almost fully solvable by unique amounts
(286/308), which leaves little for the ML stage to prove. This generator keeps
the same "two languages" (vocabulary borrowed from the provided files) but adds
the things that make real reconciliation hard:

* **non-unique amounts** - ATM withdrawals in round sums, bi-weekly payroll of
  an identical amount, repeated transfers, coffee at the same price, round
  fuel top-ups;
* **noise** - 0-5 day lags with rare outliers, cent-level rounding
  differences, register typos, mis-labelled descriptions, "Unclear
  transaction" entries;
* **one-sided entries** - bank-only fees and register-only outstanding cheques;
* optional **format drift** - from a given month the bank changes its
  descriptor format (e.g. "KROGER #1234" -> "KROGER STORE 1234 HOUSTON TX"),
  the scenario the paper argues rules-based systems break on.

Ground truth is returned explicitly; ids carry no information.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# category -> (bank formats, drifted bank formats, register phrasings, amount sampler key, type)
CATALOG = {
    "Grocery": (["KROGER #{n}", "SAFEWAY #{n}", "WHOLE FOODS #{n}", "TRADER JOES"],
                ["KROGER STORE {n} HOUSTON TX", "SAFEWAY STORE {n}", "WFM {n} AUSTIN", "TRADER JOE'S #{n}"],
                ["Groceries", "Grocery store", "Weekly groceries", "Food shopping"], "grocery", "D", 10),
    "Restaurant": (["CAFE #{n}", "RESTAURANT #{n}", "BISTRO #{n}", "DINER #{n}"],
                   ["SQ *CAFE {n}", "TST* RESTAURANT {n}", "SQ *BISTRO {n}", "TST* DINER {n}"],
                   ["Dinner out", "Restaurant", "Lunch", "Eating out"], "restaurant", "D", 9),
    "Coffee": (["STARBUCKS #{n}", "PEETS COFFEE #{n}"], ["SBUX {n} SEATTLE", "PEETS {n}"],
               ["Coffee", "Morning coffee", "Coffee shop"], "coffee", "D", 8),
    "Gas Station": (["BP GAS #{n}", "SHELL OIL #{n}", "EXXON #{n}", "CHEVRON #{n}"],
                    ["BP#{n} FUEL", "SHELL SERVICE {n}", "EXXONMOBIL {n}", "CHEVRON {n} FUEL"],
                    ["Fill up", "Fuel", "Gas", "Gas station"], "fuel", "D", 5),
    "Online Purchase": (["AMAZON.COM", "EBAY PURCHASE", "ONLINE ORDER #{n}"],
                        ["AMZN MKTP US*{n}", "EBAY O*{n}", "PAYPAL *ORDER {n}"],
                        ["Online purchase", "Amazon order", "Online shopping"], "online", "D", 6),
    "Atm Withdrawal": (["ATM WITHDRAWAL #{n}", "ATM WD {n}"], ["CASH WDL ATM {n}", "ATM CASH {n}"],
                       ["ATM cash", "Cash withdrawal", "Cash", "Withdrew cash"], "atm", "D", 4),
    "Transfer": (["ONLINE TRANSFER", "XFER TO SAVINGS"], ["ONLINE BANKING TRANSFER TO SAV", "TRANSFER TO SAV {n}"],
                 ["Transfer", "Move to savings", "Moved to savings", "Internal transfer"], "transfer", "D", 3),
    "Payroll": (["DIRECT DEP PAYROLL", "ACH CREDIT SALARY"], ["PAYROLL ACME CORP DIRDEP", "ACME CORP PPD SALARY"],
                ["Salary", "Paycheck", "Monthly salary"], "payroll", "C", 0),
    "Check Payment": (["CHECK #{chk}", "CK #{chk}"], ["CHECK {chk}", "CHK {chk}"],
                      ["Check #{chk}", "Paid by check", "Check payment"], "check", "D", 2),
    "Utility": (["ONLINE PMT ELEC CO", "ONLINE PMT WATER", "ONLINE PMT GAS CO"],
                ["ELEC CO ONLINE PMT", "CITY WATER WEB PMT", "GAS CO WEB PMT"],
                ["Electric bill", "Water bill", "Gas bill", "Utilities"], "utility", "D", 0),
    "Subscription": (["NETFLIX", "SPOTIFY", "GYM MEMBERSHIP"], ["NETFLIX.COM", "SPOTIFY USA", "PLANET FIT CLUB FEES"],
                     ["Streaming service", "Music subscription", "Gym"], "subscription", "D", 0),
    "Insurance": (["AUTO INS", "HEALTH INS PMT"], ["GEICO AUTO", "BCBS HEALTH PREM"],
                  ["Car insurance", "Health insurance", "Insurance payment"], "insurance", "D", 0),
    "Bank Fee": (["MONTHLY FEE", "SERVICE CHARGE"], ["MAINTENANCE FEE", "SVC CHG"],
                 ["Bank fee", "Monthly charge"], "fee", "D", 0),
    "Rent": (["ONLINE PMT RENT"], ["ZELLE TO LANDLORD"], ["Rent"], "rent", "D", 0),
}

FIXED = {  # recurring fixed amounts (bank format index -> amount)
    "subscription": [15.49, 10.99, 38.17], "insurance": [247.31, 412.50], "rent": [1850.00],
}


@dataclass
class SynthConfig:
    months: int = 12
    start: str = "2024-01-01"
    seed: int = 7
    volume: float = 1.0             # scales the number of discretionary transactions per month
    drift_month: int | None = 7     # 1-based month in which the bank's format changes; None = no drift
    p_round: float = 0.04           # cent-level rounding difference
    p_typo: float = 0.05
    p_mislabel: float = 0.03
    p_anomaly: float = 0.01
    p_bank_only: float = 0.015
    p_reg_only: float = 0.015
    lag_probs: tuple = (0.25, 0.32, 0.22, 0.10, 0.06, 0.03, 0.02)  # lag 0..5, then outlier bucket


def _amount(kind: str, rng: np.random.Generator, idx: int, payroll: float) -> float:
    if kind == "grocery":
        return round(float(rng.lognormal(4.4, 0.5)), 2)
    if kind == "restaurant":
        return round(float(rng.lognormal(3.7, 0.6)), 2)
    if kind == "coffee":
        return float(rng.choice([4.50, 5.25, 5.75, 6.40]))
    if kind == "fuel":
        return float(rng.choice([40.0, 50.0, 60.0])) if rng.random() < 0.4 else round(float(rng.uniform(30, 80)), 2)
    if kind == "online":
        return round(float(rng.lognormal(3.8, 0.8)), 2)
    if kind == "atm":
        return float(rng.choice([20.0, 40.0, 60.0, 100.0, 200.0]))
    if kind == "transfer":
        return float(rng.choice([100.0, 200.0, 250.0, 500.0]))
    if kind == "payroll":
        return payroll
    if kind == "check":
        return round(float(rng.lognormal(5.0, 0.7)), 2) if rng.random() < 0.7 else float(rng.choice([50.0, 100.0, 150.0]))
    if kind == "utility":
        return round(float(rng.uniform(40, 180)), 2)
    if kind in FIXED:
        return FIXED[kind][idx % len(FIXED[kind])]
    if kind == "fee":
        return [12.00, 2.50][idx % 2]
    raise ValueError(kind)


def _typo(s: str, rng: np.random.Generator) -> str:
    if len(s) < 5:
        return s
    i = int(rng.integers(1, len(s) - 1))
    return s[:i] + s[i + 1:]


def generate(cfg: SynthConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, set[tuple[str, str]]]:
    """Returns (bank_df, register_df, truth) in the same CSV schema as the provided files, plus a ``month`` column."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    start = pd.Timestamp(cfg.start)
    bank, reg, truth = [], [], set()
    chk = 3000
    k = 0
    payroll = 2450.00
    cats = list(CATALOG)
    for m in range(cfg.months):
        m_start = start + pd.DateOffset(months=m)
        days = (m_start + pd.DateOffset(months=1) - m_start).days
        drift = cfg.drift_month is not None and m + 1 >= cfg.drift_month
        events = []  # (category, format idx, day)
        for c in cats:
            bf, _, _, kind, _, per_month = CATALOG[c]
            for _ in range(int(rng.poisson(per_month * cfg.volume)) if per_month else 0):
                events.append((c, int(rng.integers(len(bf))), int(rng.integers(days))))
        events += [("Payroll", 0, 0), ("Payroll", 0, 14)]
        events += [("Subscription", i, 3 + 7 * i) for i in range(3)]
        events += [("Insurance", i, 10 + i) for i in range(2)]
        events += [("Rent", 0, 1), ("Bank Fee", 0, days - 1)]
        events += [("Utility", i, 18 + i) for i in range(3)]
        for c, fi, day in events:
            bf, bf_drift, rp, kind, tcode, _ = CATALOG[c]
            k += 1
            chk += kind == "check"
            amount = _amount(kind, rng, fi, payroll)
            n = int(rng.integers(1000, 9999))
            fmt = (bf_drift if drift else bf)[fi]
            bdesc = fmt.format(n=n, chk=chk)
            if c in ("Online Purchase",) and "{n}" not in fmt and rng.random() < 0.5:
                bdesc = bdesc.lower()  # mixed-case artefacts, as in the provided file ("bp gas #7420")
            r_date = m_start + pd.Timedelta(days=day)
            lag_bucket = int(rng.choice(len(cfg.lag_probs), p=np.array(cfg.lag_probs) / sum(cfg.lag_probs)))
            lag = lag_bucket if lag_bucket < 6 else int(rng.integers(6, 12))
            b_date = r_date + pd.Timedelta(days=lag)
            rdesc = rp[int(rng.integers(len(rp)))].format(chk=chk)
            category = c
            notes = ""
            if rng.random() < cfg.p_typo:
                rdesc = _typo(rdesc, rng)
            if rng.random() < cfg.p_mislabel:
                rdesc = CATALOG[cats[int(rng.integers(len(cats)))]][2][0].format(chk=chk)
            if rng.random() < cfg.p_anomaly:
                rdesc, category, notes = "Other expense", "Miscellaneous", "Unclear transaction"
            r_amount = amount
            if rng.random() < cfg.p_round:
                r_amount = round(amount + float(rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])) / 100, 2)
            bid, rid = f"B{k:05d}", f"R{k:05d}"
            include_b = rng.random() >= cfg.p_reg_only
            include_r = rng.random() >= cfg.p_bank_only
            if include_b:
                bank.append({"transaction_id": bid, "date": b_date, "description": bdesc, "amount": amount,
                             "type": "CREDIT" if tcode == "C" else "DEBIT", "month": m + 1})
            if include_r:
                reg.append({"transaction_id": rid, "date": r_date, "description": rdesc, "amount": r_amount,
                            "type": "CR" if tcode == "C" else "DR", "category": category, "notes": notes, "month": m + 1})
            if include_b and include_r:
                truth.add((bid, rid))
    bdf = pd.DataFrame(bank).sort_values(["date", "transaction_id"]).reset_index(drop=True)
    rdf = pd.DataFrame(reg).sort_values(["date", "transaction_id"]).reset_index(drop=True)
    # shuffle ids so that they carry no signal
    perm_b = rng.permutation(len(bdf))
    perm_r = rng.permutation(len(rdf))
    bmap = {old: f"B{i:05d}" for old, i in zip(bdf["transaction_id"], perm_b)}
    rmap = {old: f"R{i:05d}" for old, i in zip(rdf["transaction_id"], perm_r)}
    bdf["transaction_id"] = bdf["transaction_id"].map(bmap)
    rdf["transaction_id"] = rdf["transaction_id"].map(rmap)
    truth = {(bmap[b], rmap[r]) for b, r in truth}
    bdf["date"] = bdf["date"].dt.strftime("%Y-%m-%d")
    rdf["date"] = rdf["date"].dt.strftime("%Y-%m-%d")
    return bdf, rdf, truth
