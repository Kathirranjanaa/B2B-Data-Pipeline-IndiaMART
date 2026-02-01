from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # stable non-GUI backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

# =========================================================
# SETTINGS
# =========================================================
INPUT_CSV = "clean_data.csv"
PLOTS_DIR = "plots"

DPI = 160                         # slightly sharper
PNG_COMPRESS_LEVEL = 1            # fast PNG encoding
SCATTER_MAX_POINTS = 5000         # downsample for speed
TOP_N = 10

# Layout engine (best fix for cut-off labels WITHOUT bbox_inches="tight")
USE_CONSTRAINED_LAYOUT = True

plt.rcParams.update({
    # clean look
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,

    # typography
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,

    # clarity (your old code disabled this -> text looked rough)
    "text.antialiased": True,
    "lines.antialiased": True,

    # stability / speed
    "path.simplify": True,
    "agg.path.chunksize": 12000,
    "text.usetex": False,
})

def safe_subplots_adjust(fig, **kwargs):
    """Avoid matplotlib warnings: don't call subplots_adjust when constrained layout is enabled."""
    try:
        if hasattr(fig, "get_constrained_layout") and fig.get_constrained_layout():
            return
    except Exception:
        pass
    safe_subplots_adjust(fig, **kwargs)


# Warm up font cache (prevents font scan lag on Windows)
from matplotlib import font_manager
_ = font_manager.findfont("DejaVu Sans")


# =========================================================
# HELPERS
# =========================================================
def ensure_dirs() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

def safe_to_numeric(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def _short_label(s: str, max_len: int = 18) -> str:
    s = str(s).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"

def format_inr(x) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"₹{float(x):,.0f}"
    except Exception:
        return "NA"

def new_fig_ax(figsize=(10, 5)):
    # Constrained layout fixes most cutoffs (legend/xticks/title)
    if USE_CONSTRAINED_LAYOUT:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def _safe_savefig(path: str, dpi: int = DPI) -> None:
    """
    Save PNG safely with retries + temp file.
    Avoid bbox_inches="tight" to prevent Windows hangs.
    """
    save_kwargs = {
        "dpi": dpi,
        "format": "png",
        "facecolor": "white",
        "pil_kwargs": {"compress_level": PNG_COMPRESS_LEVEL, "optimize": False},
    }

    fig = plt.gcf()
    tmp_path = path + ".tmp"

    for _ in range(3):
        try:
            fig.savefig(tmp_path, **save_kwargs)
            plt.close(fig)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
            os.replace(tmp_path, path)
            return
        except (PermissionError, OSError):
            time.sleep(0.6)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise

    fig.savefig(tmp_path, **save_kwargs)
    plt.close(fig)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass
    os.replace(tmp_path, path)

def save_chart(filename: str) -> None:
    out_path = os.path.join(PLOTS_DIR, filename)
    _safe_savefig(out_path)
    print("✅ Saved:", f"{PLOTS_DIR}/{filename}")

def placeholder_chart(title: str, filename: str, msg: str = "Not enough data / column missing") -> None:
    fig, ax = new_fig_ax(figsize=(9, 4))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=16, weight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.38, msg, ha="center", va="center", fontsize=12, transform=ax.transAxes)
    save_chart(filename)


# =========================================================
# 8) TREEMAP (pure matplotlib)
# =========================================================
def treemap_slice_and_dice(labels, sizes, rect=(0, 0, 1, 1)):
    total = float(sum(sizes)) if sum(sizes) > 0 else 1.0
    x, y, w, h = rect
    rects = []
    horizontal = True

    for lab, s in zip(labels, sizes):
        frac = float(s) / total if total > 0 else 0.0
        if horizontal:
            rw = w * frac
            rects.append((x, y, rw, h, lab, s))
            x += rw
            w -= rw
        else:
            rh = h * frac
            rects.append((x, y, w, rh, lab, s))
            y += rh
            h -= rh

        total -= float(s)
        if total <= 0:
            break
        horizontal = not horizontal

    return rects


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    ensure_dirs()

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"❌ Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    safe_to_numeric(df, "price_numeric")
    safe_to_numeric(df, "rating")

    # -----------------------------------------------------
    # 1) KPI Cards (improved alignment using axes coords)
    # -----------------------------------------------------
    fig, ax = new_fig_ax(figsize=(13, 3.2))
    ax.axis("off")

    total_rows = len(df)
    uniq_suppliers = int(df["supplier_name"].nunique()) if "supplier_name" in df.columns else 0
    uniq_cities = int(df["city"].nunique()) if "city" in df.columns else 0
    uniq_states = int(df["state"].nunique()) if "state" in df.columns else 0
    median_price = df["price_numeric"].median() if "price_numeric" in df.columns else np.nan
    miss_price = (df["price_numeric"].isna().mean() * 100) if "price_numeric" in df.columns else np.nan
    miss_rating = (df["rating"].isna().mean() * 100) if "rating" in df.columns else np.nan

    kpis = [
        ("Products", f"{total_rows:,}"),
        ("Suppliers", f"{uniq_suppliers:,}"),
        ("Cities", f"{uniq_cities:,}"),
        ("States", f"{uniq_states:,}"),
        ("Median Price", format_inr(median_price)),
        ("Missing Price %", f"{miss_price:.1f}%" if not pd.isna(miss_price) else "NA"),
        ("Missing Rating %", f"{miss_rating:.1f}%" if not pd.isna(miss_rating) else "NA"),
    ]

    ax.text(0.5, 1.02, "KPI Summary", ha="center", va="bottom",
            fontsize=18, weight="bold", transform=ax.transAxes)

    n = len(kpis)
    left = 0.03
    right = 0.03
    gap = 0.014
    card_w = (1 - left - right - gap*(n-1)) / n
    y0, h = 0.12, 0.70

    for i, (lab, val) in enumerate(kpis):
        x0 = left + i*(card_w + gap)
        ax.add_patch(Rectangle((x0, y0), card_w, h,
                               transform=ax.transAxes,
                               facecolor="#0B5FAE", alpha=0.08,
                               edgecolor="none"))
        ax.text(x0 + 0.03, y0 + 0.48, lab, fontsize=10, weight="bold",
                transform=ax.transAxes)
        ax.text(x0 + 0.03, y0 + 0.20, val, fontsize=15, weight="bold",
                transform=ax.transAxes)

    save_chart("01_kpi_cards.png")

    # -----------------------------------------------------
    # 2) Line Chart (labels + spacing improved)
    # -----------------------------------------------------
    if "search_keyword" in df.columns and "price_numeric" in df.columns:
        g = df.dropna(subset=["search_keyword", "price_numeric"]).groupby("search_keyword")["price_numeric"].mean()
        g = g.sort_values(ascending=False).head(10)
        if not g.empty:
            fig, ax = new_fig_ax(figsize=(10.5, 5.5))
            x = np.arange(len(g))
            ax.plot(x, g.values, marker="o", linewidth=2.2)
            ax.set_xticks(x)
            ax.set_xticklabels([_short_label(v, 14) for v in g.index], rotation=28, ha="right")
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
            ax.set_title("Line: Avg Price by Top Keywords")
            ax.set_ylabel("Average Price (INR)")
            ax.set_xlabel("Keyword")
            ax.grid(True, axis="y")
            ax.grid(False, axis="x")
            safe_subplots_adjust(fig, bottom=0.22)  # safety (even with constrained_layout)
            save_chart("02_line.png")
        else:
            placeholder_chart("Line Chart", "02_line.png")
    else:
        placeholder_chart("Line Chart", "02_line.png")

    # -----------------------------------------------------
    # 3) Bar Chart (value labels + no clipping)
    # -----------------------------------------------------
    if "city" in df.columns:
        vc = df["city"].astype(str).value_counts().head(10)
        fig, ax = new_fig_ax(figsize=(10.5, 5.5))
        labels = [_short_label(x, 16) for x in vc.index]
        bars = ax.bar(labels, vc.values)
        ax.set_title("Bar: Top Cities by Listings")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        ax.set_axisbelow(True)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", rotation=28)
        ax.bar_label(bars, padding=2, fontsize=9)
        safe_subplots_adjust(fig, bottom=0.22)
        save_chart("03_bar.png")
    else:
        placeholder_chart("Bar Chart", "03_bar.png")

    # -----------------------------------------------------
    # 4) Donut Chart (legend fixed: bottom, not cut)
    # -----------------------------------------------------
    if "search_keyword" in df.columns:
        counts = df["search_keyword"].astype(str).value_counts()
        top = counts.head(5)
        others = counts.iloc[5:].sum()

        labels = [_short_label(x, 18) for x in top.index]
        values = top.values.astype(float).tolist()
        if others > 0:
            labels.append("Others")
            values.append(float(others))

        if sum(values) <= 0:
            placeholder_chart("Donut / Pie Chart", "04_donut.png")
        else:
            fig, ax = new_fig_ax(figsize=(10, 6.2))
            wedges, _ = ax.pie(values, startangle=90, labels=None,
                               wedgeprops={"linewidth": 1})
            centre = plt.Circle((0, 0), 0.62, fc="white")
            ax.add_artist(centre)

            total = sum(values)
            legend_text = [f"{lab} — {v/total*100:.1f}%" for lab, v in zip(labels, values)]
            ax.legend(wedges, legend_text, title="Keywords",
                      loc="upper center", bbox_to_anchor=(0.5, -0.05),
                      ncol=2, frameon=False)

            ax.set_title("Donut: Keyword Share")
            safe_subplots_adjust(fig, bottom=0.18)
            save_chart("04_donut.png")
    else:
        placeholder_chart("Donut / Pie Chart", "04_donut.png")

    # -----------------------------------------------------
    # 5) Histogram
    # -----------------------------------------------------
    if "price_numeric" in df.columns:
        p = df["price_numeric"].dropna()
        if len(p) > 0:
            p99 = p.quantile(0.99)
            p = p[p <= p99]

            fig, ax = new_fig_ax(figsize=(10.5, 5.3))
            ax.hist(p.values, bins=30)
            ax.set_title("Histogram: Price Distribution (clipped at 99%)")
            ax.set_xlabel("Price (INR)")
            ax.set_ylabel("Frequency")
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
            ax.grid(True, axis="y")
            ax.grid(False, axis="x")
            save_chart("05_hist.png")
        else:
            placeholder_chart("Histogram", "05_hist.png", "No numeric prices available")
    else:
        placeholder_chart("Histogram", "05_hist.png")

    # -----------------------------------------------------
    # 6) Map Chart
    # -----------------------------------------------------
    lat_col = None
    lon_col = None
    for a, b in [("latitude", "longitude"), ("lat", "lon"), ("lat", "lng")]:
        if a in df.columns and b in df.columns:
            lat_col, lon_col = a, b
            break

    if lat_col and lon_col:
        d = df.dropna(subset=[lat_col, lon_col]).copy()
        if not d.empty:
            fig, ax = new_fig_ax(figsize=(10.5, 5.3))
            ax.scatter(d[lon_col], d[lat_col], alpha=0.55, s=12, linewidths=0)
            ax.set_title("Map: Supplier Points (Lat/Lon)")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            save_chart("06_map.png")
        else:
            placeholder_chart("Map Chart", "06_map.png")
    elif "city" in df.columns:
        city_counts = df["city"].astype(str).value_counts().head(25)
        fig, ax = new_fig_ax(figsize=(10.5, 5.3))
        ax.scatter(range(len(city_counts)), city_counts.values, s=26)
        ax.set_title("Map Approx: City Index vs Listings")
        ax.set_xlabel("City Index (Top 25)")
        ax.set_ylabel("Listings Count")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        save_chart("06_map.png")
    else:
        placeholder_chart("Map Chart", "06_map.png")

    # -----------------------------------------------------
    # 7) Combo Chart (FIX: make line visible using twinx safely)
    # -----------------------------------------------------
    if "search_keyword" in df.columns and "price_numeric" in df.columns:
        d = df.dropna(subset=["search_keyword"]).copy()
        topk = d["search_keyword"].value_counts().head(8)

        avgp = df.dropna(subset=["search_keyword", "price_numeric"]).groupby("search_keyword")["price_numeric"].mean()
        avgp = avgp.reindex(topk.index)

        if not topk.empty:
            fig, ax = new_fig_ax(figsize=(11, 5.7))
            x = np.arange(len(topk))

            bars = ax.bar(x, topk.values, label="Listings Count", alpha=0.90)
            ax.set_ylabel("Listings Count")
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
            ax.grid(True, axis="y")
            ax.grid(False, axis="x")

            ax2 = ax.twinx()
            ax2.plot(x, avgp.values, marker="o", linewidth=2.2, label="Avg Price (INR)")
            ax2.set_ylabel("Avg Price (INR)")
            ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

            ax.set_xticks(x)
            ax.set_xticklabels([_short_label(k, 14) for k in topk.index], rotation=28, ha="right")

            ax.set_title("Combo: Listings (Bar) + Avg Price (Line)")
            ax.bar_label(bars, padding=2, fontsize=9)

            # clean combined legend
            lines, labels_ = ax.get_legend_handles_labels()
            lines2, labels2_ = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels_ + labels2_, frameon=False, loc="upper left")

            safe_subplots_adjust(fig, bottom=0.22)
            save_chart("07_combo.png")
        else:
            placeholder_chart("Combo Chart", "07_combo.png")
    else:
        placeholder_chart("Combo Chart", "07_combo.png")

    # -----------------------------------------------------
    # 8) Treemap (better text contrast + spacing)
    # -----------------------------------------------------
    if "search_keyword" in df.columns:
        s = df["search_keyword"].value_counts().head(8)
        labels = [_short_label(x, 14) for x in s.index]
        sizes = s.values.astype(float).tolist()

        rects = treemap_slice_and_dice(labels, sizes, rect=(0, 0, 1, 1))

        fig, ax = new_fig_ax(figsize=(11, 5.5))
        ax.set_axis_off()
        ax.set_title("Treemap: Top Keywords by Count")

        for (x0, y0, w, h, lab, val) in rects:
            ax.add_patch(Rectangle((x0, y0), w, h, facecolor="#0B5FAE", alpha=0.18, edgecolor="black", lw=0.8))
            ax.text(x0 + w * 0.03, y0 + h * 0.60, f"{lab}\n{int(val)}",
                    fontsize=10, weight="bold")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        save_chart("08_treemap.png")
    else:
        placeholder_chart("Treemap", "08_treemap.png")

    # -----------------------------------------------------
    # 9) Waterfall (insight-driven)
    # -----------------------------------------------------
    # Use TOP 10 most expensive items (more meaningful than "first 10 rows")
    if "price_numeric" in df.columns:
        priced = df[["product_name", "price_numeric"]].copy() if "product_name" in df.columns else df[["price_numeric"]].copy()
        priced["price_numeric"] = pd.to_numeric(priced["price_numeric"], errors="coerce")
        priced = priced.dropna(subset=["price_numeric"]).sort_values("price_numeric", ascending=False).head(10)

        if len(priced) > 0:
            vals = priced["price_numeric"].values.astype(float)
            cum = np.cumsum(vals)

            labels = None
            if "product_name" in priced.columns:
                labels = priced["product_name"].astype(str).fillna("")
                labels = labels.apply(lambda x: (x[:18] + "…") if len(x) > 19 else x).tolist()

            fig, ax = new_fig_ax(figsize=(10.5, 5.3))
            ax.bar(range(len(vals)), vals, alpha=0.85)
            ax.plot(range(len(vals)), cum, marker="o", linewidth=2.2)

            ax.set_title("Waterfall: Top 10 Prices + Cumulative")
            ax.set_xlabel("Top 10 most expensive items")
            ax.set_ylabel("Price (INR)")
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
            ax.grid(True, axis="y")
            ax.grid(False, axis="x")

            if labels:
                ax.set_xticks(range(len(vals)))
                ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)

            save_chart("09_waterfall.png")
        else:
            placeholder_chart("Waterfall", "09_waterfall.png")
    else:
        placeholder_chart("Waterfall", "09_waterfall.png")
# 10) Scatter (clearer points + axes)
    # -----------------------------------------------------
    if "rating" in df.columns and "price_numeric" in df.columns:
        sc = df.dropna(subset=["rating", "price_numeric"]).copy()
        if not sc.empty:
            p99 = sc["price_numeric"].quantile(0.99)
            sc = sc[sc["price_numeric"] <= p99]

            if len(sc) > SCATTER_MAX_POINTS:
                sc = sc.sample(n=SCATTER_MAX_POINTS, random_state=42)

            fig, ax = new_fig_ax(figsize=(10.5, 5.3))
            ax.scatter(sc["rating"], sc["price_numeric"], alpha=0.55, s=14, marker="o", linewidths=0)
            ax.set_title("Scatter: Rating vs Price (clipped at 99%, sampled)")
            ax.set_xlabel("Rating")
            ax.set_ylabel("Price (INR)")
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
            ax.grid(True, axis="y")
            ax.grid(False, axis="x")
            save_chart("10_scatter.png")
        else:
            placeholder_chart("Scatter Plot", "10_scatter.png")
    else:
        placeholder_chart("Scatter Plot", "10_scatter.png")

    print("\n✅ DONE: Exactly 10 PNG charts saved in ./plots\n")


if __name__ == "__main__":
    main()