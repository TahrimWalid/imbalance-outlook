"""Figures: forecast vs actual imbalance price, with the p10-p90 band.

    python -m imbalance figures runs/validation/xgb/forecast_backtest.csv

Writes PNGs to runs/figures/. Contains spot prices (Nord Pool licence): for local use and
the jury presentation only, never publish the files.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C

# reference palette (validated light mode: blue / orange / aqua, all pairs pass)
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
FORECAST, ACTUAL, SPOT = "#2a78d6", "#eb6834", "#1baf7a"
TZ = "Europe/Helsinki"
# forecast origins shown in the 10-hour panels: fixed in advance (spread over weeks and times of day),
# not picked by looking at how good the forecast was
ORIGINS_LOCAL = ["2026-09-10 06:00", "2026-09-17 14:00", "2026-09-24 18:00", "2026-10-01 22:00"]
WEEK_LOCAL = ("2026-09-14", "2026-09-21")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _clip_marks(ax, x, y, top):
    """Values above the axis are drawn as markers on the top edge with their value."""
    over = y > top
    for xi, yi in zip(x[over], y[over]):
        ax.annotate(f"{yi:,.0f} ↑", (xi, top), xytext=(0, -2), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color=ACTUAL)


def load(forecast: Path, actuals: Path) -> pd.DataFrame:
    f = pd.read_csv(forecast, parse_dates=["forecast_origin_utc", "delivery_start_utc"])
    a = pd.read_csv(actuals, parse_dates=["delivery_start_utc"])
    return f.merge(a[["delivery_start_utc", "imbalance_price", "spot_price"]], on="delivery_start_utc", how="left")


def ten_hour_panels(d: pd.DataFrame, out: Path, title: str, top: float = 400) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharey=True, facecolor=SURFACE)
    sel = d[d["forecast_origin_utc"].isin([pd.Timestamp(x, tz=TZ).tz_convert("UTC") for x in ORIGINS_LOCAL])]
    vals = sel[["p10", "p90", "imbalance_price", "spot_price"]].to_numpy().ravel()
    lo_y, hi_y = float(pd.Series(vals).min()), min(float(pd.Series(vals).max()), top)
    pad = 0.06 * (hi_y - lo_y)
    lo_y, top = lo_y - pad, hi_y + pad
    for ax, o_local in zip(axes.flat, ORIGINS_LOCAL):
        o = pd.Timestamp(o_local, tz=TZ).tz_convert("UTC")
        g = d[d["forecast_origin_utc"] == o].sort_values("step")
        h = (g["step"].to_numpy() - 1) / 4                 # hours ahead
        _style(ax)
        ax.fill_between(h, g["p10"], g["p90"], color=FORECAST, alpha=0.15, linewidth=0, label="Forecast range p10–p90")
        ax.plot(h, g["spot_price"], color=SPOT, lw=1.5, ls=(0, (4, 2)), label="Spot price (known in advance)")
        ax.plot(h, g["p50"], color=FORECAST, lw=2, marker="o", ms=3.5, label="Forecast p50")
        ax.plot(h, g["imbalance_price"].clip(upper=top), color=ACTUAL, lw=2, marker="o", ms=3.5,
                label="Actual imbalance price (Fingrid)")
        _clip_marks(ax, h, g["imbalance_price"].to_numpy(), top)
        ax.set_title(f"Forecast made {pd.Timestamp(o_local):%a %d %b %H:%M} (Finnish time)", fontsize=10.5,
                     color=INK, loc="left")
        ax.set_xlim(0, 10)
        ax.set_ylim(lo_y, top)
        ax.set_xticks(range(0, 11, 2))
        inside = ((g["imbalance_price"] >= g["p10"]) & (g["imbalance_price"] <= g["p90"])).mean()
        mae = (g["imbalance_price"] - g["p50"]).abs().mean()
        ax.text(0.99, 0.97, f"avg miss €{mae:.0f} · inside range {inside:.0%}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=INK2)
    for ax in axes[1]:
        ax.set_xlabel("Hours ahead", color=INK2, fontsize=9.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("€/MWh", color=INK2, fontsize=9.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(title, fontsize=14, color=INK, x=0.06, ha="left", y=0.99)
    fig.text(0.06, 0.955, "Next 10 hours from four forecast times (fixed in advance). Practice exam period, 7 Sep – 4 Oct 2026.",
             fontsize=9.5, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    path = out / "forecast_10h_panels.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def week_view(d: pd.DataFrame, out: Path, title: str, top: float = 400, lead: int = 1) -> Path:
    """Each quarter-hour as forecast `lead` hours ahead (steps 4*(lead-1)+1 .. 4*lead: one forecast per quarter-hour)."""
    lo, hi = (pd.Timestamp(x, tz=TZ).tz_convert("UTC") for x in WEEK_LOCAL)
    s0, s1 = 4 * (lead - 1) + 1, 4 * lead
    g = d[d["step"].between(s0, s1) & (d["delivery_start_utc"] >= lo) & (d["delivery_start_utc"] < hi)].sort_values("delivery_start_utc")
    t = g["delivery_start_utc"].dt.tz_convert(TZ)
    fig, ax = plt.subplots(figsize=(13, 4.8), facecolor=SURFACE)
    _style(ax)
    ax.fill_between(t, g["p10"].clip(upper=top), g["p90"].clip(upper=top), color=FORECAST, alpha=0.15, linewidth=0,
                    label="Forecast range p10–p90")
    ax.plot(t, g["p50"].clip(upper=top), color=FORECAST, lw=1.4, label="Forecast p50")
    fover = g["p50"] > top
    if fover.any():
        ax.plot(t[fover], [top] * int(fover.sum()), ls="none", marker="v", ms=6, color=FORECAST, clip_on=False)
    ax.plot(t, g["imbalance_price"].clip(upper=top), color=ACTUAL, lw=1.4, label="Actual imbalance price (Fingrid)")
    over = g["imbalance_price"] > top
    if over.any():
        ax.plot(t[over], [top] * int(over.sum()), ls="none", marker="^", ms=7, color=ACTUAL, clip_on=False)
        ax.text(0.995, 0.97, f"▲ actual above €{top:.0f}: {int(over.sum())} quarter-hours (highest €{g['imbalance_price'].max():,.0f})"
                + (f"   ▼ forecast above €{top:.0f}: {int(fover.sum())}" if fover.any() else ""),
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    ax.set_ylim(top=top)
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=TZ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=TZ))
    ax.set_ylabel("€/MWh", color=INK2, fontsize=9.5)
    inside = ((g["imbalance_price"] >= g["p10"]) & (g["imbalance_price"] <= g["p90"])).mean()
    mae = (g["imbalance_price"] - g["p50"]).abs().mean()
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title, x=0.06, ha="left", fontsize=13, color=INK)
    ahead = "up to 1 hour before it starts" if lead == 1 else f"{lead - 1}–{lead} hours before it starts"
    fig.text(0.06, 0.905, f"One week (14–21 Sep), each quarter-hour as forecast {ahead} · "
             f"avg miss €{mae:.0f} · inside range {inside:.0%}", fontsize=9.5, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    path = out / f"forecast_week_{lead}h.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def main(forecast: Path, title: str = "Imbalance price: forecast vs actual") -> list[Path]:
    out = C.RUNS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    d = load(forecast, C.DATA_DIR / "actuals_validation.csv")
    return [ten_hour_panels(d, out, title)] + [week_view(d, out, title, lead=h) for h in (1, 5, 10)]
