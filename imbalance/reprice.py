"""Reproduce Fingrid's published imbalance price (319) from its inputs, and re-price old quarter-hours.

    python -m imbalance reprice reproduce      -> runs/pricing/reproduction.md (+ csv details)

Phase 3 of the extended-history work: which datasets, units and timestamps make imbalance/pricing.py
reproduce 319. The day-ahead price is not used yet (no licensed-free source until the ENTSO-E fetch),
so quarter-hours without a dominating direction are reported separately and not scored.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import pricing as P

EXT_DIR = C.ROOT / "data_ext"
OUT_DIR = C.RUNS_DIR / "pricing"

# columns of the wide frame (series_id in data_ext/inputs.parquet)
IMB, DIR = "fi_imbalance_price", "fi_mfrr_direction"
VWA_UP, VWA_DOWN = "fi_afrr_up_price", "fi_afrr_down_price"
MP_UP, MP_DOWN = "fi_mfrr_up_price", "fi_mfrr_down_price"
SPOT = "fi_spot_price"


def load_wide(data_dir: Path = EXT_DIR) -> pd.DataFrame:
    raw = pd.read_parquet(data_dir / "inputs.parquet", columns=["series_id", "timestamp_utc", "value"])
    raw = raw[raw["series_id"].str.startswith("fi_")]
    w = raw.pivot(index="timestamp_utc", columns="series_id", values="value")
    w.index = pd.DatetimeIndex(w.index, tz="UTC") if w.index.tz is None else w.index
    spot = data_dir / "spot_entsoe.csv"           # ENTSO-E day-ahead price FI (not for re-use; local only)
    if spot.is_file():
        sp = pd.read_csv(spot, usecols=["timestamp_utc", "value"])
        w[SPOT] = pd.Series(sp["value"].to_numpy(), index=pd.to_datetime(sp["timestamp_utc"], utc=True))
    return w.sort_index()


@dataclass(frozen=True)
class Candidate:
    """One reading of the open questions: which datasets are the two volumes, and their timestamp shift."""
    name: str
    afrr_up: tuple[str, ...]          # summed (absolute values), MW average over the quarter-hour
    afrr_down: tuple[str, ...]
    mfrr_up: tuple[str, ...]
    mfrr_down: tuple[str, ...]
    afrr_scale: float = 1.0           # relative unit factor aFRR vs mFRR (both MW -> MWh cancels)
    shift_afrr: int = 0               # quarter-hours; +1 = use the value stamped one quarter later
    shift_mfrr: int = 0


def _vol(w: pd.DataFrame, cols: tuple[str, ...], shift: int) -> np.ndarray:
    if not cols:
        return np.zeros(len(w))
    v = sum(w[c].abs().fillna(0.0) if c in w else 0.0 for c in cols)
    v = v.shift(-shift) if shift else v
    return np.asarray(v, float)


def price_frame(w: pd.DataFrame, cand: Candidate | None, rule: str) -> pd.DataFrame:
    """Price under `rule` with the inputs of `cand` (cand None: old rule, no volumes)."""
    kw = {}
    if rule == "new":
        kw = dict(afrr_vol_up=cand.afrr_scale * _vol(w, cand.afrr_up, cand.shift_afrr),
                  afrr_vol_down=cand.afrr_scale * _vol(w, cand.afrr_down, cand.shift_afrr),
                  mfrr_vol_up=_vol(w, cand.mfrr_up, cand.shift_mfrr),
                  mfrr_vol_down=_vol(w, cand.mfrr_down, cand.shift_mfrr))
    out = P.price(w[DIR].to_numpy(), w[VWA_UP], w[VWA_DOWN], w[MP_UP], w[MP_DOWN],
                  day_ahead=w[SPOT] if SPOT in w else np.full(len(w), np.nan), rule=rule, **kw)
    out.index = w.index
    out["published"] = w[IMB]
    out["direction"] = w[DIR]
    out["diff"] = out["price"] - out["published"]
    return out


def score(pf: pd.DataFrame, tol: float = 0.005 + 1e-9) -> pd.DataFrame:
    """Per case: rows, share within 0.01 EUR/MWh, mean and max |diff| (day-ahead case not scored)."""
    rows = []
    for case, g in pf[pf["published"].notna()].groupby("case"):
        d = g["diff"].abs()
        rows.append({"case": case, "quarter_hours": len(g), "scored": int(d.notna().sum()),
                     "share_within_0.01": float((d <= 0.01 + 1e-9).mean()) if d.notna().any() else np.nan,
                     "share_exact": float((d <= tol).mean()) if d.notna().any() else np.nan,
                     "mean_abs_diff": float(d.mean()), "max_abs_diff": float(d.max()),
                     "mean_diff": float(g["diff"].mean())})
    return pd.DataFrame(rows).set_index("case")


NEW_RULE_START = pd.Timestamp("2026-05-31T21:00:00Z")          # 1 Jun 2026 00:00 Finnish time
QH_PRICE_START = pd.Timestamp("2025-03-18T23:00:00Z")          # first quarter-hour-priced hour (data)

AFRR = {   # Finland's satisfied aFRR demand, candidates (up, down)
    "349": (("fi_afrr_up_vol_marginal",), ("fi_afrr_down_vol_marginal",)),
    "349+354": (("fi_afrr_up_vol_marginal", "fi_afrr_up_vol_local"), ("fi_afrr_down_vol_marginal", "fi_afrr_down_vol_local")),
    "349+354+407": (("fi_afrr_up_vol_marginal", "fi_afrr_up_vol_local", "fi_afrr_up_avoided"),
                    ("fi_afrr_down_vol_marginal", "fi_afrr_down_vol_local", "fi_afrr_down_avoided")),
    "349+407": (("fi_afrr_up_vol_marginal", "fi_afrr_up_avoided"), ("fi_afrr_down_vol_marginal", "fi_afrr_down_avoided")),
}
MFRR = {   # mFRR energy activated in Finland, candidates (up, down)
    "375": (("fi_mfrr_up_activated",), ("fi_mfrr_down_activated",)),
    "385+390": (("fi_mfrr_up_sa", "fi_mfrr_up_da"), ("fi_mfrr_down_sa", "fi_mfrr_down_da")),
    "385": (("fi_mfrr_up_sa",), ("fi_mfrr_down_sa",)),
    "375+391": (("fi_mfrr_up_activated", "fi_mfrr_special_up"), ("fi_mfrr_down_activated", "fi_mfrr_special_down")),
}


def candidates(shifts=(-1, 0, 1)) -> list[Candidate]:
    out = []
    for an, (au, ad) in AFRR.items():
        for mn, (mu, md) in MFRR.items():
            for sa in shifts:
                for sm in shifts:
                    out.append(Candidate(f"aFRR {an}{sa:+d} / mFRR {mn}{sm:+d}", au, ad, mu, md,
                                         shift_afrr=sa, shift_mfrr=sm))
    return out


def implied_afrr_share(w: pd.DataFrame) -> pd.DataFrame:
    """Quarter-hours where the published price lies strictly inside (VWA, mFRR price): the weighted case
    with both components. There P = (v*a + m*M)/(a + M), so a/(a+M) = (m - P)/(m - v) follows from the
    published price alone, without any volume dataset."""
    d = w[DIR]
    v = np.where(d == 1, w[VWA_UP], np.where(d == -1, w[VWA_DOWN], np.nan))
    m = np.where(d == 1, w[MP_UP], np.where(d == -1, w[MP_DOWN], np.nan))
    p = w[IMB].to_numpy()
    lo, hi = np.fmin(v, m), np.fmax(v, m)
    inside = (p > lo + 0.02) & (p < hi - 0.02) & (hi - lo > 1.0)
    share = (m - p) / (m - v)
    return pd.DataFrame({"direction": d, "vwa": v, "mfrr_price": m, "published": p,
                         "implied_afrr_share": share}, index=w.index)[inside]


# --------------------------------------------------------------------------------------------------
# Chosen inputs (settled by the reproduction, runs/pricing/reproduction.md)
# --------------------------------------------------------------------------------------------------
CHOSEN = Candidate("aFRR 349/350 + mFRR 385+390 / 386+389",
                   ("fi_afrr_up_vol_marginal",), ("fi_afrr_down_vol_marginal",),
                   ("fi_mfrr_up_sa", "fi_mfrr_up_da"), ("fi_mfrr_down_sa", "fi_mfrr_down_da"))


def hourly_old_rule(w: pd.DataFrame) -> pd.DataFrame:
    """Old rule with hourly prices (before 19 Mar 2025): hourly direction and mFRR price, and the hourly VWA
    aFRR as the 349/350-weighted mean of the quarter-hour VWA values (reproduces Fingrid's Q&A example of
    10.3.2025 23:00 EET exactly: 233.00)."""
    H = w.index.floor("h")

    def vwa(pc, vc):
        v = w[vc].abs().where(w[pc].notna())
        num, den = (w[pc] * v).groupby(H).sum(min_count=1), v.groupby(H).sum(min_count=1)
        return (num / den).where(den > 0)

    g = pd.DataFrame({"published": w[IMB].groupby(H).first(), "direction": w[DIR].groupby(H).first(),
                      "vwa_up": vwa(VWA_UP, "fi_afrr_up_vol_marginal"), "vwa_down": vwa(VWA_DOWN, "fi_afrr_down_vol_marginal"),
                      "mfrr_up": w[MP_UP].groupby(H).first(), "mfrr_down": w[MP_DOWN].groupby(H).first(),
                      "day_ahead": (w[SPOT] if SPOT in w else pd.Series(np.nan, index=w.index)).groupby(H).first()})
    pr = P.price(g["direction"].to_numpy(), g["vwa_up"], g["vwa_down"], g["mfrr_up"], g["mfrr_down"],
                 day_ahead=g["day_ahead"], rule="old")
    pr.index = g.index
    out = pr.join(g[["published", "direction"]])
    out["diff"] = out["price"] - out["published"]
    return out


def classify_old(pf: pd.DataFrame) -> pd.Series:
    """Reason for each old-rule mismatch (> 0.01 EUR/MWh)."""
    t = pf.index
    reason = pd.Series("unexplained", index=pf.index, dtype=object)
    excl = (pf["published"] - pf["mfrr_price"]).abs() <= 0.01 + 1e-9
    reason[excl] = "aFRR component excluded by Fingrid (319 = mFRR price)"
    for a, b, text in C.PRICE_EXCEPTIONS:
        inside = (t >= pd.Timestamp(a)) & (t <= pd.Timestamp(b))
        reason[inside & excl] = "known Fingrid correction: " + text.split(":")[0]
    off1 = ((pf["published"] - pf["mfrr_price"]).abs().round(2).isin([1.0, 2.0])
            | (pf["published"] - pf["vwa"]).abs().round(2).isin([1.0, 2.0]))
    reason[~excl & off1] = "one component differs from 319 by a whole euro (component data revised after the price)"
    missing = pf["mfrr_price"].isna() | (pf["vwa"].isna() & (pf["published"] < pf["mfrr_price"]) & (pf["direction"] == 1)) \
        | (pf["vwa"].isna() & (pf["published"] < pf["mfrr_price"]) & (pf["direction"] == -1))
    reason[~excl & ~off1 & missing] = "missing input (component price not published)"
    pic = (t >= pd.Timestamp("2025-03-26T22:00:00Z")) & (t < pd.Timestamp("2025-03-28T22:00:00Z"))
    reason[pic & (reason == "unexplained")] = "PICASSO accession days 27-28.3.2025"
    return reason


def classify_new(pf: pd.DataFrame, w: pd.DataFrame) -> pd.Series:
    d = pf["direction"]
    da = np.where(d == 1, w["fi_mfrr_up_da"].fillna(0), w["fi_mfrr_down_da"].fillna(0)) != 0
    reason = pd.Series("unexplained", index=pf.index, dtype=object)
    excl = ((pf["published"] - pf["mfrr_price"]).abs() <= 0.01 + 1e-9).to_numpy()
    reason[~da & excl] = "aFRR component excluded by Fingrid (319 = mFRR price)"
    reason[da] = "direct activation (DA) in the dominating direction: 389/390 give the maximum power of the " \
                 "quarter-hour, not its energy, so the mFRR weight is too high"
    return reason


def _md_table(df: pd.DataFrame, floatfmt: int = 2) -> str:
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else f"{v:.{floatfmt}f}" if abs(v) < 1e6 else f"{v:.3g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def _within(diff: pd.Series) -> float:
    d = diff.abs().dropna()
    return float((d <= 0.01 + 1e-6).mean()) if len(d) else np.nan


def reproduce(data_dir: Path = EXT_DIR, out_dir: Path = OUT_DIR, log=print) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    w = load_wide(data_dir)
    end = w.index.max()
    new = w[NEW_RULE_START:end]
    old = w[QH_PRICE_START:NEW_RULE_START - C.Q]
    hourly = w[C.EXT_START:QH_PRICE_START - C.Q]
    md = []

    # ---- Step 1: candidates on the new-rule period
    log("step 1: candidate grid")
    grid = []
    for cand in candidates():
        pf = price_frame(new, cand, "new")
        pf = pf[pf["direction"].isin([1, -1])]
        wt = pf[pf["case"] == "weighted"]
        grid.append({"candidate": cand.name, "directional within 0.01": _within(pf["diff"]),
                     "weighted quarter-hours": len(wt), "weighted within 0.01": _within(wt["diff"]),
                     "weighted mean abs diff": float(wt["diff"].abs().mean()),
                     "marginal within 0.01": _within(pf.loc[pf["case"] == "marginal", "diff"])})
    grid = pd.DataFrame(grid).sort_values("directional within 0.01", ascending=False).set_index("candidate")
    grid.to_csv(out_dir / "step1_candidates.csv")

    pf1 = price_frame(new, CHOSEN, "new")
    pf1["n_case"] = pf1["case"]
    sc1 = score(pf1[pf1["direction"].isin([1, -1])])
    bad1 = pf1[pf1["direction"].isin([1, -1]) & (pf1["diff"].abs() > 0.01 + 1e-6)].copy()
    bad1["reason"] = classify_new(bad1, new.loc[bad1.index])
    inputs = ["fi_afrr_up_price", "fi_afrr_down_price", "fi_mfrr_up_price", "fi_mfrr_down_price",
              "fi_afrr_up_vol_marginal", "fi_afrr_down_vol_marginal", "fi_mfrr_up_sa", "fi_mfrr_down_sa",
              "fi_mfrr_up_da", "fi_mfrr_down_da", "fi_mfrr_special_up", "fi_mfrr_special_down"]
    worst1 = bad1.assign(abs_diff=bad1["diff"].abs()).sort_values("abs_diff", ascending=False).head(20)
    worst1 = worst1[["direction", "case", "published", "price", "diff", "reason"]].join(new[inputs])
    bad1.join(new[inputs]).to_csv(out_dir / "step1_mismatches.csv")
    no_dir_new = int((new[DIR] == 0).sum())
    have_spot = SPOT in w
    nd_rows = []
    if have_spot:
        for nm, part in (("12.6.2024-18.3.2025 (hourly)", w[C.EXT_START:QH_PRICE_START - C.Q]),
                         ("19.3.2025-31.5.2026", w[QH_PRICE_START:NEW_RULE_START - C.Q]), ("from 1.6.2026", new)):
            x = part[part[DIR] == 0]
            dd = (x[IMB] - x[SPOT]).dropna()
            nd_rows.append({"period": nm, "quarter-hours": len(x), "with spot": len(dd),
                            "319 = day-ahead (within 0.01)": float((dd.abs() <= 0.01 + 1e-6).mean()) if len(dd) else np.nan,
                            "mean diff": dd.mean(), "p5": dd.quantile(.05), "median": dd.median(),
                            "p95": dd.quantile(.95), "max abs": dd.abs().max()})
    nd_table = pd.DataFrame(nd_rows).set_index("period") if nd_rows else None

    # ---- Step 2: old rule, quarter-hour prices, then hourly prices
    log("step 2: old rule")
    pf2 = price_frame(old, None, "old")
    d2 = pf2[pf2["direction"].isin([1, -1])]
    sc2 = score(d2)
    bad2 = d2[d2["diff"].abs() > 0.01 + 1e-6].copy()
    bad2["reason"] = classify_old(bad2)
    bad2.join(old[inputs[:4]]).to_csv(out_dir / "step2_mismatches.csv")
    worst2 = bad2.assign(abs_diff=bad2["diff"].abs()).sort_values("abs_diff", ascending=False).head(20)
    worst2 = worst2[["direction", "published", "price", "diff", "vwa", "mfrr_price", "reason"]]
    reasons2 = bad2.groupby("reason").size().rename("quarter-hours").to_frame()
    days2 = bad2.groupby([bad2.index.tz_convert("Europe/Helsinki").date, "reason"]).size().rename("quarter-hours").to_frame()
    days2.index = [f"{a} | {b}" for a, b in days2.index]
    days2.index.name = "local date | reason"
    no_dir_old = int((old[DIR] == 0).sum())

    hr = hourly_old_rule(hourly)
    hd = hr[hr["direction"].isin([1, -1])]
    hbad = hd[hd["diff"].abs() > 0.01 + 1e-6]
    h_missing_dir = int(hr["direction"].isna().sum())

    # ---- Step 3: the volumes before and after 1 Jun 2026
    log("step 3: volume continuity")
    M = w.index.tz_convert("Europe/Helsinki").strftime("%Y-%m")
    vols = {"349 aFRR up": "fi_afrr_up_vol_marginal", "350 aFRR down": "fi_afrr_down_vol_marginal",
            "385 SA up": "fi_mfrr_up_sa", "386 SA down": "fi_mfrr_down_sa", "390 DA up": "fi_mfrr_up_da", "389 DA down": "fi_mfrr_down_da"}
    cont = {}
    for k, c in vols.items():
        s = w[c].abs()
        cont[f"{k}: active share"] = (s > 0).groupby(M).mean()
        cont[f"{k}: median MW when active"] = s.where(s > 0).groupby(M).median()
    cont = pd.DataFrame(cont)
    cont.index.name = "month"
    cont.to_csv(out_dir / "step3_volumes_by_month.csv")
    consistency = []
    for nm, part in (("12.6.2024-18.3.2025", hourly), ("19.3.2025-31.5.2026", old), ("from 1.6.2026", new)):
        for pc, vc in ((VWA_UP, "fi_afrr_up_vol_marginal"), (VWA_DOWN, "fi_afrr_down_vol_marginal")):
            consistency.append({"period": nm, "pair": f"{pc} vs {vc}",
                                "share where VWA present <=> volume > 0": float((part[pc].notna() == (part[vc].abs() > 0)).mean())})
    consistency = pd.DataFrame(consistency).set_index("period")

    # ---- Step 4: bound of the re-pricing error
    log("step 4: error bound")
    pn, po = price_frame(old, CHOSEN, "new"), pf2
    d = old[DIR].to_numpy()
    lo, hi = P.bounds(d, old[VWA_UP], old[VWA_DOWN], old[MP_UP], old[MP_DOWN])
    gap = pd.Series(hi - lo, index=old.index)
    wt = pn["case"] == "weighted"
    exact = wt & (pn["vwa"].isna() | pn["afrr_vol"].fillna(0).eq(0))
    both = wt & ~exact
    da = wt & (np.where(d == 1, old["fi_mfrr_up_da"].fillna(0), old["fi_mfrr_down_da"].fillna(0)) != 0)
    q = [.25, .5, .75, .9, .95, .99]
    gaps = pd.DataFrame({"all weighted with both components": gap[both].quantile(q),
                         "of which DA in the dominating direction": gap[both & da].quantile(q)}).T
    gaps.columns = [f"p{int(x * 100)}" for x in q]
    gaps["max"] = [gap[both].max(), gap[both & da].max()]
    gaps["n"] = [int(both.sum()), int((both & da).sum())]
    gaps.index.name = "quarter-hours"
    change = (pn["price"] - po["price"])[wt]
    pd.DataFrame({"gap": gap, "case_new": pn["case"], "repriced": pn["price"], "old_rule": po["price"],
                  "published": old[IMB]}).to_csv(out_dir / "step4_gap.csv")

    nd = new.loc[pf1.index]
    dd = pf1["direction"]
    no_da = (np.where(dd == 1, nd["fi_mfrr_up_da"].fillna(0), nd["fi_mfrr_down_da"].fillna(0)) == 0) & dd.isin([1, -1]).to_numpy()
    two = ((nd["fi_mfrr_up_activated"].fillna(0) != 0) & (nd["fi_mfrr_down_activated"].fillna(0) != 0)).to_numpy()
    miss = (pf1["diff"].abs() > 0.01 + 1e-6).to_numpy()
    both_stats = (f"{int((miss & no_da & two).sum())} of {int((no_da & two).sum())} quarter-hours with Finnish mFRR in "
                  f"both directions do not match, against {int((miss & no_da & ~two).sum())} of {int((no_da & ~two).sum())} "
                  "with one direction; those few are listed under Step 1.")

    # ---- gates
    w_share = _within(pf1.loc[pf1["case"] == "weighted", "diff"])
    o_share = _within(d2["diff"])
    gate1 = w_share >= 0.98
    gate2 = o_share >= 0.98
    unexplained1 = int((bad1["reason"] == "unexplained").sum())
    unexplained2 = int((bad2["reason"] == "unexplained").sum())

    md += [
        "# Reproducing Fingrid's imbalance price (dataset 319)",
        "",
        f"Built by `python -m imbalance reprice` from `data_ext/` (Fingrid open data, {C.EXT_START:%Y-%m-%d} .. {end:%Y-%m-%d %H:%M}Z). "
        "Prices in EUR/MWh; a quarter-hour matches when |computed - published| <= 0.01. The day-ahead price is "
        "not used yet (the licence-free ENTSO-E source is not fetched yet), so quarter-hours without a dominating "
        "direction are counted but not scored.",
        "",
        "## Gates",
        "",
        f"- Step 1, new rule on new data, weighted case: **{w_share:.2%}** within 0.01 (gate 98%): "
        f"**{'PASS' if gate1 else 'FAIL'}**; mismatches without an explanation: {unexplained1}.",
        f"- Step 2, old rule on 19.3.2025-31.5.2026, quarter-hours with a direction: **{o_share:.2%}** "
        f"(gate 98%): **{'PASS' if gate2 else 'FAIL'}**; mismatches without an explanation: {unexplained2}.",
        f"- Step 2b, old rule with hourly prices, 12.6.2024-18.3.2025: {_within(hd['diff']):.2%} of {hd['diff'].notna().sum()} "
        f"hours with a direction (reported separately; not re-priced).",
        "",
        "## Readings of Fingrid's documents",
        "",
        "- New rule (Imbalance price in Finland, 1.6.2026): \"As of 1 June 2026, the imbalance price will be calculated "
        "on a volume-weighted basis when activated mFRR in Finland is in the dominating direction. The imbalance price "
        "is calculated by using the 15-minute volume-weighted price of aFRR, the marginal price of mFRR, the satisfied "
        "demand of aFRR from the PICASSO platform, and the mFRR energy activated in Finland.\" Volumes are in MWh in the "
        "formula. Reading: both volumes are energies; Fingrid's MW datasets are quarter-hour averages, and only the "
        "ratio of the two volumes matters, so MW can be used directly when both are averages (confirmed by the data, "
        "except for DA, see below).",
        "- \"If the activated mFRR in Finland is in the opposite direction than the dominating direction\" the old "
        "max/min applies. The document does not mention a quarter-hour with Finnish mFRR in both directions. Reading: "
        "only the volume in the dominating direction counts (if it is > 0 the price is weighted). The data support "
        f"it: without DA, {both_stats}",
        "- \"The result will be rounded to two decimal (0,01 EUR/MWh).\" The rounding mode is not stated; "
        "round-half-away-from-zero is used. No mismatch in this report is a 0.01 difference in the quarter-hour "
        "periods, so the mode is not decisive.",
        "- Fingrid's Q&A (28.8.2025) converts SA power to energy as \"aktivoitu alas 115 MW eli 28,75 MWh\" "
        "(activated down 115 MW, i.e. 28.75 MWh): MW x 0.25 h, so SA datasets are quarter-hour average power.",
        "- The 29.8.2024 document: \"The volume-weighted average price of aFRR energy for the hour, and in the future "
        "for the fifteen minutes, determines the imbalance price\" and \"The imbalance price is the same for every "
        "quarter of an hour in the hour until the 15- minute mFRR and ID markets.\"",
        "- Q&A 3: \"Erikoissäätö ei aseta mFRR-energian, eikä siten tasepoikkeaman hintaa\" (special regulation does "
        "not set the mFRR price, nor therefore the imbalance price). Special-regulation volumes (391/392) are "
        "therefore excluded; adding them makes the reproduction worse (table below).",
        "",
        "## Step 1: new rule on new data (from 1 Jun 2026)",
        "",
        "Candidates: aFRR satisfied demand = 349/350 (activation volume with marginal price), +353/354 (local "
        "selection), +407/408 (avoided activation); mFRR energy in Finland = 375/376 (SA+DA balancing), 385/386 (SA "
        "only), 385+390 / 386+389 (SA+DA, identical to 375/376 in every quarter-hour), 375+391 (with special "
        "regulation); timestamp shifts of -1, 0, +1 quarter-hour for each volume. All timestamps are the API's "
        "startTime (period start), with the period length from endTime. Best 12 and worst 3 of 144:",
        "",
        _md_table(pd.concat([grid.head(12), grid.tail(3)]), 4),
        "",
        f"Chosen: **{CHOSEN.name}**, no shift. 375/376 are exactly 385+390 / 386+389 in every quarter-hour of the "
        "window, so the two names are the same quantity.",
        "",
        "Per case (chosen inputs, quarter-hours with a direction):",
        "",
        _md_table(sc1, 4),
        "",
        f"Quarter-hours without a dominating direction (day-ahead case): {no_dir_new}"
        + (" (scored with the ENTSO-E day-ahead price, see 'No dominating direction')." if have_spot else
           " (not scored: no day-ahead price yet)."),
        "",
        "Mismatches by reason:",
        "",
        _md_table(bad1.groupby("reason").size().rename("quarter-hours").to_frame()),
        "",
        f"Mean signed difference of the mismatches {bad1['diff'].mean():.2f} (computed minus published): the computed "
        "price leans towards the mFRR price, as an overstated mFRR volume would make it. In every mismatch with DA, "
        "solving the published price for the DA energy gives a fraction of max power x 0.25 h between 0 and 1 "
        "(about half are 1, the rest correspond to 1-14 minutes of activation): the inputs are right, only the DA "
        "energy is unknown from a maximum-power dataset. Dataset 342 (1-min activated mFRR) is a net value of SA and "
        "DA in both directions and cannot isolate DA energy.",
        "",
        "20 worst quarter-hours (all inputs):",
        "",
        _md_table(worst1),
        "",
        "## No dominating direction: the day-ahead price",
        "",
        ("Fingrid: \"If there is no dominating mFRR direction in the market time unit, the imbalance price will be "
         "based on the Day-ahead price, even if aFRR activations have been made during that time period.\" "
         "Published price minus the ENTSO-E day-ahead price FI (same coupled price as Nord Pool's), quarter-hours "
         "with direction 0:") if have_spot else "No day-ahead price available (spot_entsoe.csv missing).",
        "",
        _md_table(nd_table, 3) if nd_table is not None else "",
        "",
        "## Step 2: old rule on old data",
        "",
        "### 19 Mar 2025 - 31 May 2026 (quarter-hour prices)",
        "",
        _md_table(sc2, 4),
        "",
        f"Quarter-hours without a dominating direction: {no_dir_old} (not scored). The 'missing' case is a quarter-hour "
        "with a direction and neither component published.",
        "",
        "Mismatches by reason:",
        "",
        _md_table(reasons2),
        "",
        "By local date:",
        "",
        _md_table(days2),
        "",
        "20 worst:",
        "",
        _md_table(worst2),
        "",
        "Notes: the 23 Oct 2025 exclusion (01:00-11:45 Finnish time) is confirmed by the data (319 = mFRR price). "
        "The same pattern (319 = mFRR price although VWA aFRR exists) also occurs on 22 Apr 2026 and on a few single "
        "days; no Fingrid announcement was found for those. Fingrid's 5.9.2025 notice of a sign error in VWA aFRR "
        "(6 May - 25 Aug 2025) says the error was corrected 'in eSett's data'; open data 319 and 347/348 agree with "
        "each other throughout that period.",
        "",
        "### 12 Jun 2024 - 18 Mar 2025 (hourly prices)",
        "",
        f"Hourly direction and mFRR price; hourly VWA aFRR = 349/350-weighted mean of the quarter-hour 347/348 values "
        f"(the simple mean matches far fewer hours). Hours with a direction: {len(hd)}, within 0.01: "
        f"{_within(hd['diff']):.2%}, mismatches: {len(hbad)} (all with a missing aFRR price in the dominating "
        f"direction while 319 is below the mFRR price). Dataset 369 is missing for {h_missing_dir} hours "
        "between 12 Jun and 8 Aug 2024, which cannot be scored; 319 has unrounded values until 2024-08-07.",
        "",
        "## Step 3: the volumes before 1 Jun 2026",
        "",
        "The old rule does not use the volumes, so they are checked indirectly.",
        "",
        "- 349/350 have the same description, unit (MW) and full coverage from 12 Jun 2024; VWA aFRR is present "
        "exactly when the 349/350 volume is > 0 (table). The hourly-period reproduction weights 347/348 by 349/350 "
        "and matches 319 (e.g. Fingrid's own example of 10.3.2025: 233.00), which shows 349/350 were the weighting "
        "volume of VWA aFRR in 2024-2025 too.",
        "- 385/386/389/390 start on 4 Mar 2025 (mFRR EAM go-live), before the re-pricing window (19 Mar 2025). "
        "No definition change is documented; the monthly distributions below show no break at 1 Jun 2026. The "
        "share of quarter-hours with up-SA rises from about 0.4-0.5 in 2025 to 0.7-0.98 from Nov 2025, and "
        "aFRR volumes are higher from Jan 2026; both are gradual market changes, not breaks at a regime date.",
        "",
        _md_table(consistency, 4),
        "",
        _md_table(cont.round(2)),
        "",
        "The re-pricing window does not need to be shortened.",
        "",
        "## Step 4: how wrong can a re-priced quarter-hour be?",
        "",
        f"19.3.2025-31.5.2026: {int(np.isin(d, [1, -1]).sum())} quarter-hours with a direction; {int(wt.sum())} "
        f"({wt.sum() / np.isin(d, [1, -1]).sum():.1%}) are 'weighted' under the new rule. {int(exact.sum())} of them are exact "
        "whatever the volumes (one component missing or zero aFRR volume). For the others the re-priced value lies "
        "between the two component prices, so the gap is the largest possible error. Only DA quarter-hours have an "
        "uncertain volume (Step 1); elsewhere the volumes reproduce 319 exactly.",
        "",
        _md_table(gaps),
        "",
        f"Re-priced minus old-rule price over the weighted quarter-hours: changed in {(change.abs() > 0.005).mean():.1%}; "
        f"p1 {change.quantile(.01):.2f}, p10 {change.quantile(.1):.2f}, median {change.median():.2f}, "
        f"p90 {change.quantile(.9):.2f}, p99 {change.quantile(.99):.2f}.",
        "",
        "## Mismatch classes",
        "",
        "- known Fingrid correction (23 Oct 2025), aFRR component excluded on other days (319 = mFRR price)",
        "- missing input (a component price not published)",
        "- component data revised after the price was published (319 differs from a component by exactly 1 or 2 EUR)",
        "- PICASSO accession days (27-28 Mar 2025)",
        "- DA volume: maximum power, not energy (new rule only)",
        "- special regulation: excluded by Fingrid's rule; including it makes the match worse",
        "- both-direction activation: no separate class needed (see readings)",
        "- our bug: none found",
        "",
    ]
    path = out_dir / "reproduction.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"wrote {path}")
    return path


QUALITY_MARK = "\n## Coverage by regime (python -m imbalance reprice)\n"


def quality_by_regime(data_dir: Path = EXT_DIR) -> str:
    """Coverage of every series in each regime period, first/last value and source resolutions; appended to
    data_ext/quality_report.md (replacing an earlier copy of this section)."""
    raw = pd.read_parquet(data_dir / "inputs.parquet", columns=["series_id", "dataset_id", "timestamp_utc", "value"])
    edges = [r.start_utc for r in C.REGIMES] + [raw["timestamp_utc"].max() + C.Q]
    labels = [f"{a:%Y-%m-%d}" for a in edges[:-1]]
    q = pd.read_csv(data_dir / "quality_report.csv", dtype=str)
    res = {}
    for r in q.dropna(subset=["source_info"]).itertuples():
        try:
            import json
            res[r.series_id] = json.loads(r.source_info).get("source_durations_min", {})
        except ValueError:
            pass
    rows = []
    for sid, g in raw.groupby("series_id"):
        g = g.dropna(subset=["value"])
        row = {"series_id": sid, "dataset": g["dataset_id"].iloc[0] if len(g) else "",
               "first": f"{g['timestamp_utc'].min():%Y-%m-%d %H:%M}" if len(g) else "",
               "last": f"{g['timestamp_utc'].max():%Y-%m-%d %H:%M}" if len(g) else "",
               "source periods (min: rows)": ", ".join(f"{k}: {v}" for k, v in sorted(res.get(sid, {}).items(), key=lambda x: int(x[0])))}
        for lab, a, b in zip(labels, edges[:-1], edges[1:]):
            n = int(((b - a) / C.Q))
            row[f"cover from {lab}"] = len(g[(g["timestamp_utc"] >= a) & (g["timestamp_utc"] < b)]) / n
        rows.append(row)
    t = pd.DataFrame(rows).set_index("series_id")
    regimes = "\n".join(f"- {r.start_utc:%Y-%m-%d %H:%M}Z: {r.name}. Source: {r.source}" for r in C.REGIMES)
    text = (QUALITY_MARK + "\nRegime periods (imbalance/config.py REGIMES):\n\n" + regimes + "\n\n"
            "Share of quarter-hours with a value in each regime period. Sparse series (aFRR prices, avoided "
            "activation) have rows only when there was activation, so a low share is not a gap.\n\n"
            + _md_table(t, 2) + "\n")
    path = data_dir / "quality_report.md"
    body = path.read_text(encoding="utf-8")
    body = body.split(QUALITY_MARK)[0].rstrip("\n") + "\n"
    path.write_text(body + text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------------------------------
# Phase 4: the re-priced series fi_imbalance_price_v2
# --------------------------------------------------------------------------------------------------
V2 = "fi_imbalance_price_v2"
V2_FILE = "imbalance_v2.parquet"


def build_v2(w: pd.DataFrame, fu: pd.Series) -> pd.DataFrame:
    """Imbalance price as if the 1 Jun 2026 rule had always applied (from 19 Mar 2025).

    from 1 Jun 2026        Fingrid's published 319
    19.3.2025-31.5.2026    published 319, except where the new rule's case is 'weighted', there is no direct
                           activation (DA) in the dominating direction (DA energy is unknown, published price
                           kept) and the old rule reproduces the published price (so Fingrid's manual
                           corrections stay as published): there the new formula's value
    before 19.3.2025       published 319 (hourly prices; for lag features only, never a training label)
    first_usable_origin_utc = that of 319 for every quarter-hour.
    """
    old = w[QH_PRICE_START:NEW_RULE_START - C.Q]
    pn, po = price_frame(old, CHOSEN, "new"), price_frame(old, None, "old")
    d = old[DIR].to_numpy()
    da = np.where(d == 1, old["fi_mfrr_up_da"].fillna(0), np.where(d == -1, old["fi_mfrr_down_da"].fillna(0), 0)) != 0
    reproduces = (po["diff"].abs() <= 0.01 + 1e-6).to_numpy()
    weighted = (pn["case"] == "weighted").to_numpy()
    rep = weighted & ~da & reproduces
    out = pd.DataFrame({"value": w[IMB], "published": w[IMB], "case": "published", "repriced": False}, index=w.index)
    out.loc[old.index, "case"] = np.where(rep, "weighted", np.where(weighted & da, "weighted_da_kept",
                                          np.where(weighted & ~reproduces, "weighted_correction_kept", pn["case"])))
    out.loc[old.index[rep], "value"] = pn["price"].to_numpy()[rep]
    out.loc[old.index[rep], "repriced"] = True
    out.loc[w.index < QH_PRICE_START, "case"] = "hourly_not_repriced"
    out.loc[w.index >= NEW_RULE_START, "case"] = "new_rule_published"
    out["direction"] = w[DIR]
    lo, hi = P.bounds(w[DIR].to_numpy(), w[VWA_UP], w[VWA_DOWN], w[MP_UP], w[MP_DOWN])
    out["low"], out["high"] = lo, hi
    out["first_usable_origin_utc"] = fu.reindex(w.index)
    return out[out["published"].notna()]


def check_v2(v2: pd.DataFrame) -> dict:
    """Invariants on the real series: a re-priced value is <= published when up, >= when down, and lies
    between the two component prices."""
    r = v2[v2["repriced"]]
    up, dn = r["direction"] == 1, r["direction"] == -1
    checks = {
        "re-priced <= published (direction up)": bool((r.loc[up, "value"] <= r.loc[up, "published"] + 0.01).all()),
        "re-priced >= published (direction down)": bool((r.loc[dn, "value"] >= r.loc[dn, "published"] - 0.01).all()),
        "re-priced within component prices": bool(((r["value"] >= r["low"] - 0.01) & (r["value"] <= r["high"] + 0.01)).all()),
        "only 19.3.2025-31.5.2026 re-priced": bool(((r.index >= QH_PRICE_START) & (r.index < NEW_RULE_START)).all()),
        "first_usable equals 319's": bool(v2["first_usable_origin_utc"].notna().all()),
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise AssertionError(f"fi_imbalance_price_v2 invariants failed: {bad}")
    return checks


def write_v2(data_dir: Path = EXT_DIR, out_dir: Path = OUT_DIR, log=print) -> Path:
    raw = pd.read_parquet(data_dir / "inputs.parquet")
    fu = raw[raw["series_id"] == IMB].set_index("timestamp_utc")["first_usable_origin_utc"]
    fu.index = pd.DatetimeIndex(fu.index, tz="UTC") if fu.index.tz is None else fu.index
    w = load_wide(data_dir)
    v2 = build_v2(w, pd.to_datetime(fu, utc=True))
    checks = check_v2(v2)
    path = data_dir / V2_FILE
    keep = v2[["value", "published", "case", "repriced", "first_usable_origin_utc"]].rename_axis("timestamp_utc").reset_index()
    keep.to_parquet(path, index=False)
    log(f"wrote {path} ({int(v2['repriced'].sum())} quarter-hours re-priced)")

    # how much the history changed
    out_dir.mkdir(parents=True, exist_ok=True)
    win = v2[(v2.index >= QH_PRICE_START) & (v2.index < NEW_RULE_START)]
    ch = win["value"] - win["published"]
    q = [.01, .05, .1, .25, .5, .75, .9, .95, .99]
    lines = ["# fi_imbalance_price_v2: how re-pricing changed the history", "",
             "19 Mar 2025 - 31 May 2026. Built by `python -m imbalance reprice --v2` (rule in "
             "`imbalance/reprice.py:build_v2`).", "", "Invariants on the real series (all must hold):", ""]
    lines += [f"- {k}: {'yes' if v else 'NO'}" for k, v in checks.items()]
    lines += ["", "Cases:", "", _md_table(win.groupby("case").size().rename("quarter-hours").to_frame()), "",
              f"Changed quarter-hours: {int((ch.abs() > 0.005).sum())} of {len(win)} ({(ch.abs() > 0.005).mean():.1%}); "
              f"mean change {ch.mean():.2f} EUR/MWh, mean absolute change {ch.abs().mean():.2f}.", "",
              "Change (re-priced minus published) over the re-priced quarter-hours, percentiles:", "",
              _md_table(ch[win["repriced"]].quantile(q).rename(lambda x: f"p{int(round(x * 100))}").to_frame("EUR/MWh").T), ""]
    if SPOT in w:
        sp = w[SPOT].reindex(win.index)
        rows = []
        for nm, s in (("published", win["published"]), ("re-priced", win["value"])):
            prem = s - sp
            rows.append({"series": nm, "high spikes (> spot + 50)": float((prem > C.SPIKE_MARGIN).mean()),
                         "low spikes (< spot - 50)": float((prem < -C.SPIKE_MARGIN).mean()),
                         "mean |price - spot|": float(prem.abs().mean())})
        lines += ["Spike frequency (share of quarter-hours with a spot price):", "",
                  _md_table(pd.DataFrame(rows).set_index("series"), 4), ""]
    (out_dir / "repricing_effect.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {out_dir / 'repricing_effect.md'}")
    return path
