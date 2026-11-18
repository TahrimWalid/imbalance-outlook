"""Figures for the post-hackathon write-up, safe to show outside the team.

    .venv/bin/python scripts/portfolio_figures.py [--variant B]

Writes PNGs (and on request a GIF) to docs/img/. Only Fingrid open data (CC BY 4.0) and our own
forecasts and scores are drawn: no spot price line, no Nord Pool data. The forecast comes from
runs/validation/combined_v3 (or runs/variants/<v>/validation/combined_v3 with --variant).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
FORECAST, ACTUAL = BLUE, ORANGE
TZ = "Europe/Helsinki"
OUT = ROOT / "docs" / "img"
CREDIT = "Source: Fingrid open data (CC BY 4.0), own forecasts."
# chosen before looking at forecast quality (same origins as imbalance/figures.py)
ORIGINS_LOCAL = ["2026-09-10 06:00", "2026-09-17 14:00", "2026-09-24 18:00", "2026-10-01 22:00"]
SIM_DAY = "2026-09-24"
RULE_DAY = "2025-09-17"
LEVEL1 = [("mae_p50", "Accuracy\nMAE of p50"), ("calibration_pinball_mean", "Range\npinball p10/p90"),
          ("brier_spike_mean", "Spikes\nBrier"), ("brier_direction", "Direction\nBrier")]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def header(fig, title, sub, y=0.985):
    fig.suptitle(title, fontsize=14, color=INK, x=0.04, ha="left", y=y)
    fig.text(0.04, y - 0.045, sub, fontsize=9.5, color=INK2, ha="left", va="top")


def credit(fig):
    fig.text(0.04, 0.012, CREDIT, fontsize=8, color=MUTED, ha="left", va="bottom")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(p)
    return p


def load_forecast(variant: str | None) -> pd.DataFrame:
    base = ROOT / "runs" / ("variants/" + variant if variant else "") / "validation" / "combined_v3"
    f = pd.read_csv(base / "forecast_backtest.csv", parse_dates=["forecast_origin_utc", "delivery_start_utc"])
    # imbalance price only: the spot column of this file is never read
    a = pd.read_csv(ROOT / "data" / "actuals_validation.csv", usecols=["delivery_start_utc", "imbalance_price"],
                    parse_dates=["delivery_start_utc"])
    return f.merge(a, on="delivery_start_utc", how="left")


# 1. the rule change on one real day ----------------------------------------------------------------

def rule_day():
    v = pd.read_parquet(ROOT / "data_ext" / "imbalance_v2.parquet")
    t = v["timestamp_utc"].dt.tz_convert(TZ)
    d = v[t.dt.strftime("%Y-%m-%d") == RULE_DAY].copy()
    d["t"] = d["timestamp_utc"].dt.tz_convert(TZ)
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=SURFACE)
    style(ax)
    ax.step(d["t"], d["published"], where="post", color=ORANGE, lw=1.8, label="Published price (old rule: most extreme of aFRR and mFRR price)")
    ax.step(d["t"], d["value"], where="post", color=AQUA, lw=1.8, label="Re-priced with the 1 Jun 2026 rule (volume-weighted)")
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3), tz=TZ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
    ax.set_ylabel("€/MWh", color=INK2)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ch = (d["value"] - d["published"]).abs()
    header(fig, "Same day, two pricing rules",
           f"{pd.Timestamp(RULE_DAY):%a %d %b %Y}, Finnish time, 15-min imbalance price. "
           f"{(ch > 0.005).mean():.0%} of quarter-hours change, by €{ch.mean():.0f}/MWh on average; "
           "the new rule weights\nthe aFRR and mFRR prices by volume, so the price sits between them and swings shrink in both directions.")
    credit(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.86))
    return save(fig, "01_rule_change_day.png")


# 2. our model against the official baseline ------------------------------------------------------

def read_summary(p: Path) -> dict:
    s = pd.read_csv(p)
    if {"metric", "value"} <= set(s.columns):
        return dict(zip(s["metric"], s["value"]))
    return s.iloc[0].to_dict()


def model_vs_baseline(variant: str | None):
    fmt = lambda x: f"{x:.1f}" if x > 1 else f"{x:.3f}"
    base = read_summary(ROOT / "baseline_output" / "validation" / "score" / "score_summary.csv")
    ours_p = ROOT / "runs" / ("variants/" + variant if variant else "") / "validation" / "combined_v3" / "score" / "score_summary.csv"
    ours = read_summary(ours_p)
    gain = [100 * (base[k] - ours[k]) / base[k] for k, _ in LEVEL1]
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    style(ax)
    x = np.arange(len(LEVEL1))
    ax.bar(x, gain, width=0.5, color=BLUE)
    for xi, g, (k, _) in zip(x, gain, LEVEL1):
        ax.text(xi, g + 1, f"{g:.0f}% better\n{fmt(base[k])} → {fmt(ours[k])}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, [lab for _, lab in LEVEL1], color=INK2)
    ax.set_ylim(0, max(gain) * 1.35)
    ax.set_ylabel("Error reduction vs official baseline (%)", color=INK2)
    header(fig, "Forecast vs the organisers' baseline",
           "Validation month 7 Sep – 4 Oct 2026, 26,700 quarter-hours scored with the official score.py. "
           "Lower error is better; bar = how much lower ours is.")
    credit(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    return save(fig, "02_model_vs_baseline.png")


# 3. four 10-hour forecasts --------------------------------------------------------------------------

def ten_hour_panels(d: pd.DataFrame, cap: float = 400):
    sel = d[d["forecast_origin_utc"].isin([pd.Timestamp(o, tz=TZ).tz_convert("UTC") for o in ORIGINS_LOCAL])]
    top = min(float(sel[["p90", "imbalance_price"]].max().max()) * 1.1, cap)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharey=True, facecolor=SURFACE)
    for ax, o_local in zip(axes.flat, ORIGINS_LOCAL):
        o = pd.Timestamp(o_local, tz=TZ).tz_convert("UTC")
        g = d[d["forecast_origin_utc"] == o].sort_values("step")
        h = (g["step"].to_numpy() - 1) / 4
        style(ax)
        ax.fill_between(h, g["p10"], g["p90"], color=FORECAST, alpha=0.15, lw=0, label="Forecast range p10–p90")
        ax.plot(h, g["p50"], color=FORECAST, lw=2, label="Forecast p50")
        y = g["imbalance_price"].to_numpy()
        ax.plot(h, np.minimum(y, top), color=ACTUAL, lw=2, marker="o", ms=3.5, label="Actual imbalance price")
        for xi, yi in zip(h[y > top], y[y > top]):
            ax.annotate(f"{yi:,.0f} ↑", (xi, top), xytext=(0, -2), textcoords="offset points", ha="center",
                        va="top", fontsize=8, color=INK2)
        ax.set_title(f"Made {pd.Timestamp(o_local):%a %d %b %H:%M}", fontsize=10.5, color=INK, loc="left")
        ax.set_xlim(0, 10)
        ax.set_xticks(range(0, 11, 2))
        inside = ((g["imbalance_price"] >= g["p10"]) & (g["imbalance_price"] <= g["p90"])).mean()
        mae = (g["imbalance_price"] - g["p50"]).abs().mean()
        ax.text(0.99, 0.97, f"avg miss €{mae:.0f} · inside range {inside:.0%}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=INK2)
    lo = min(float(d.loc[d["forecast_origin_utc"].isin([pd.Timestamp(o, tz=TZ).tz_convert("UTC") for o in ORIGINS_LOCAL]),
                         ["p10", "imbalance_price"]].min().min()), 0)
    axes[0, 0].set_ylim(lo - 20, top)
    for ax in axes[1]:
        ax.set_xlabel("Hours ahead", color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("€/MWh", color=INK2)
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper left", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.035, 0.925))
    header(fig, "Next 10 hours, four forecasts",
           "Forecast times fixed in advance, not picked by result. Validation month, Finnish time.")
    credit(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    return save(fig, "03_forecast_10h_panels.png")


# 4. simulation GIF: one day, a new forecast every hour --------------------------------------------

def sim_gif(d: pd.DataFrame):
    day0 = pd.Timestamp(SIM_DAY + " 06:00", tz=TZ)
    origins = [day0 + pd.Timedelta(hours=k) for k in range(0, 17)]
    act = d.drop_duplicates("delivery_start_utc")[["delivery_start_utc", "imbalance_price"]]
    act = act.assign(t=act["delivery_start_utc"].dt.tz_convert(TZ)).sort_values("t")
    fc = {o: d[d["forecast_origin_utc"] == o.tz_convert("UTC")].sort_values("step") for o in origins}
    win = act[(act["t"] >= day0 - pd.Timedelta(hours=6)) & (act["t"] < origins[-1] + pd.Timedelta(hours=10))]
    vals = pd.concat([win["imbalance_price"]] + [f["p90"] for f in fc.values()] + [f["p10"] for f in fc.values()])
    top = min(float(vals.max()) * 1.08, 600)
    ymin = min(float(vals.min()), 0) - 20
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.12)
    fig.text(0.04, 0.965, "Simulated live run: a new 10-hour forecast every hour", fontsize=13, color=INK, va="top")
    fig.text(0.04, 0.905, f"{day0:%a %d %b %Y}, Finnish time. Validation month, forecast made only with data "
             "published before each hour.", fontsize=9, color=INK2, va="top")
    fig.text(0.99, 0.012, CREDIT, fontsize=7.5, color=MUTED, ha="right", va="bottom")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Patch(color=FORECAST, alpha=0.25), Line2D([], [], color=FORECAST, lw=2), Line2D([], [], color=ACTUAL, lw=2)],
               ["Forecast range p10–p90", "Forecast p50", "Actual imbalance price, known so far"],
               loc="upper left", bbox_to_anchor=(0.035, 0.87), ncol=3, frameon=False, fontsize=9)
    frames = origins + [origins[-1]] * 4

    def draw(o):
        ax.clear()
        style(ax)
        lo_t, hi_t = o - pd.Timedelta(hours=6), o + pd.Timedelta(hours=10)
        known = act[(act["t"] >= lo_t) & (act["t"] < o)]
        ax.axvspan(o, hi_t, color=FORECAST, alpha=0.04, lw=0)
        ax.axvline(o, color=INK2, lw=1, ls=(0, (3, 2)))
        f = fc[o]
        t = f["delivery_start_utc"].dt.tz_convert(TZ)
        ax.fill_between(t, f["p10"].clip(upper=top), f["p90"].clip(upper=top), color=FORECAST, alpha=0.2, lw=0, step="post")
        ax.step(t, f["p50"].clip(upper=top), where="post", color=FORECAST, lw=2)
        ax.step(known["t"], known["imbalance_price"].clip(upper=top), where="post", color=ACTUAL, lw=2)
        ax.set_xlim(lo_t, hi_t)
        ax.set_ylim(ymin, top)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 2), tz=TZ))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
        ax.set_ylabel("€/MWh", color=INK2)
        ax.text(o, top, f" now {o:%H:%M}", va="top", ha="left", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(lo_t, top, " past 6 h", va="top", ha="left", fontsize=9, color=INK2)
        ax.text(hi_t, top, "next 10 h ", va="top", ha="right", fontsize=9, color=INK2)

    anim = FuncAnimation(fig, draw, frames=frames, interval=700)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "04_live_simulation.gif"
    anim.save(p, writer=PillowWriter(fps=1.4), dpi=110)
    plt.close(fig)
    print(p)
    return p


# 5. variants against A, with the day-block bootstrap ---------------------------------------------

VARIANT_LABEL = {"B": "B: 14 months, re-priced", "F": "F: B + the hourly year 2024–25, regime flag"}
VARIANT_COLOR = {"B": BLUE, "F": AQUA}


def variants_chart(variants: list[str]):
    """Change of each official metric vs A on the validation month, with the 90% day-block bootstrap interval."""
    from imbalance import variants as V
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    off = np.linspace(-0.14, 0.14, len(variants)) if len(variants) > 1 else [0.0]
    for j, v in enumerate(variants):
        a, b = V._align(V._load("A", "validation"), V._load(v, "validation"))
        r = V.bootstrap_diff(a, b, "")
        for i, (k, lab) in enumerate(LEVEL1):
            ref = r.loc[k, "A"]
            pt, lo, hi = (100 * r.loc[k, c] / ref for c in ("difference", "90% low", "90% high"))
            y = i + off[j]
            ax.plot([lo, hi], [y, y], color=VARIANT_COLOR[v], lw=2, solid_capstyle="round")
            ax.plot(pt, y, "o", ms=8, color=VARIANT_COLOR[v], mec=SURFACE, mew=2,
                    label=VARIANT_LABEL[v] if i == 0 else None)
            ax.annotate(f"{pt:+.1f}%", (lo, y), xytext=(-6, 0), textcoords="offset points", va="center", ha="right",
                        fontsize=8.5, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(left=ax.get_xlim()[0] - 1)
    ax.set_yticks(range(len(LEVEL1)), [lab.replace("\n", " · ") for _, lab in LEVEL1], color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("Change vs A (4 months of history), %   ← better", color=INK2)
    ax.legend(loc="lower left", frameon=False, fontsize=9, bbox_to_anchor=(0, 1.0), ncol=len(variants))
    header(fig, "Longer history, priced by today's rule",
           "Validation month 7 Sep – 4 Oct 2026. Dot = change in the official metric, line = 90% interval "
           "(whole days resampled).\nA line that stays left of zero is a clear improvement.", y=0.985)
    credit(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.87))
    return save(fig, "05_variants_vs_A.png")



# 6. the unseen test month: forecast one hour ahead against Fingrid's published price ------------------

def test_week(variant: str = "B", days: int = 7, name: str = "06_test_week.png"):
    f = pd.read_csv(ROOT / "runs" / "variants" / variant / "test_full" / "combined_v3" / "forecast_backtest.csv",
                    parse_dates=["delivery_start_utc"])
    f = f[f["step"] <= 4].set_index("delivery_start_utc")
    a = pd.read_csv(ROOT / "data" / "actuals_test.csv", usecols=["delivery_start_utc", "imbalance_price"],
                    parse_dates=["delivery_start_utc"]).set_index("delivery_start_utc")
    t0 = pd.Timestamp("2026-10-05", tz=TZ).tz_convert("UTC")      # first full week, fixed in advance
    g = pd.date_range(t0, t0 + pd.Timedelta(days=days) - pd.Timedelta(minutes=15), freq="15min")
    f, a = f.reindex(g), a.reindex(g)
    t = g.tz_convert(TZ)
    top = 300
    fig, ax = plt.subplots(figsize=(13, 4.8), facecolor=SURFACE)
    style(ax)
    ax.fill_between(t, f["p10"].clip(upper=top), f["p90"].clip(upper=top), step="post", color=FORECAST, alpha=0.18, lw=0, label="Likely range p10–p90")
    ax.step(t, a["imbalance_price"].clip(upper=top), where="post", color=INK2, lw=1.1, label="Actual price (Fingrid)")
    ax.step(t, f["p50"].clip(upper=top), where="post", color=FORECAST, lw=1.6, label="Forecast, made 1 h ahead")
    over = a["imbalance_price"] > top
    if over.any():
        ax.text(0.995, 0.97, f"▲ actual above €{top}: {int(over.sum())} quarter-hours (highest €{a['imbalance_price'].max():,.0f})",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    ax.set_ylim(top=top)
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=TZ) if days <= 7 else mdates.WeekdayLocator(byweekday=0, tz=TZ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=TZ))
    ax.set_ylabel("€/MWh", color=INK2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9)
    e = (a["imbalance_price"] - f["p50"]).abs()
    inside = ((a["imbalance_price"] >= f["p10"]) & (a["imbalance_price"] <= f["p90"])).mean()
    header(fig, "Unseen test month: forecast against what actually happened",
           (f"First week of the test month (5–11 Oct 2026, chosen in advance), never used for any choice. " if days <= 7 else
            f"The whole test month, 5 Oct – 1 Nov 2026, never used for any choice. Forecast made 1 hour ahead. ")
           + f"Average miss €{e.mean():.0f}/MWh, half within €{e.median():.0f}; {inside:.0%} of prices inside the range.")
    credit(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.89))
    return save(fig, name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None, help="take the forecast and scores from runs/variants/<v>")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--compare", nargs="*", default=None, help="variants to chart against A")
    a = ap.parse_args()
    want = set(a.only or ["rule", "score", "panels", "testweek"])
    if "rule" in want:
        rule_day()
    if "score" in want:
        model_vs_baseline(a.variant)
    if want & {"panels", "gif"}:
        d = load_forecast(a.variant)
        if "panels" in want:
            ten_hour_panels(d)
        if "gif" in want:
            sim_gif(d)
    if "testweek" in want:
        test_week(a.variant or "B")
        test_week(a.variant or "B", 28, "07_test_month.png")
    if a.compare:
        variants_chart(a.compare)


if __name__ == "__main__":
    main()
