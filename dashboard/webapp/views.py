import os
import numpy as np
import pandas as pd

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

# B2B_Data_Pipeline (one level above dashboard folder)
BASE_PIPELINE_DIR = os.path.dirname(settings.BASE_DIR)
CLEAN_CSV = os.path.join(BASE_PIPELINE_DIR, "clean_data.csv")


# ----------------------------
# Helpers
# ----------------------------
def load_df() -> pd.DataFrame:
    if not os.path.exists(CLEAN_CSV):
        return pd.DataFrame()

    df = pd.read_csv(CLEAN_CSV)

    # normalize numerics
    if "price_numeric" in df.columns:
        df["price_numeric"] = pd.to_numeric(df["price_numeric"], errors="coerce")
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # normalize strings (safe)
    for col in [
        "city", "state", "supplier_region", "supplier_name",
        "search_keyword", "price_bucket", "product_name"
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()

    return df


def apply_filters(df: pd.DataFrame, state: str | None, keyword: str | None) -> pd.DataFrame:
    if df.empty:
        return df

    if state and "state" in df.columns:
        df = df[df["state"].str.lower() == state.lower()]

    if keyword and "search_keyword" in df.columns:
        df = df[df["search_keyword"].str.lower() == keyword.lower()]

    return df


def safe_text(v) -> str:
    """Return clean text for JSON/template usage."""
    if v is None:
        return ""
    try:
        if isinstance(v, float) and np.isnan(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


# ----------------------------
# Pages
# ----------------------------
def home(request):
    return render(request, "home.html")


def charts(request):
    return render(request, "charts.html")


def table(request):
    df = load_df()
    if df.empty:
        return render(request, "table.html", {"columns": [], "rows": []})

    keep_cols = [c for c in [
        "product_name", "supplier_name", "city", "state",
        "supplier_region", "search_keyword", "price_numeric",
        "rating", "price_bucket"
    ] if c in df.columns]

    view = df[keep_cols].head(30)

    rows = []
    for _, row in view.iterrows():
        d = {}
        for c in keep_cols:
            val = row.get(c, None)

            if c in ("price_numeric", "rating"):
                if pd.isna(val):
                    d[c] = None
                else:
                    try:
                        d[c] = float(val)
                    except Exception:
                        d[c] = None
            else:
                d[c] = safe_text(val)

        rows.append(d)

    return render(request, "table.html", {"columns": keep_cols, "rows": rows})


# ----------------------------
# APIs
# ----------------------------
def api_filters(request):
    df = load_df()

    states = []
    keywords = []

    if not df.empty:
        if "state" in df.columns:
            states = sorted(df["state"].dropna().unique().tolist())
        if "search_keyword" in df.columns:
            keywords = sorted(df["search_keyword"].dropna().unique().tolist())

    # remove blanks/nan-like
    bad = {"", "nan", "none", "null", "unknown", "na", "n/a", "-"}
    states = [s for s in states if safe_text(s).lower() not in bad]
    keywords = [k for k in keywords if safe_text(k).lower() not in bad]

    return JsonResponse({"states": states[:300], "keywords": keywords[:300]})


def api_summary(request):
    df = load_df()
    df = apply_filters(df, request.GET.get("state"), request.GET.get("keyword"))

    total_rows = int(len(df)) if not df.empty else 0
    unique_suppliers = int(df["supplier_name"].nunique()) if "supplier_name" in df.columns and not df.empty else 0
    unique_cities = int(df["city"].nunique()) if "city" in df.columns and not df.empty else 0
    unique_states = int(df["state"].nunique()) if "state" in df.columns and not df.empty else 0

    priced = df["price_numeric"].dropna() if "price_numeric" in df.columns and not df.empty else pd.Series([], dtype=float)
    median_price = float(priced.median()) if len(priced) else 0.0
    avg_price = float(priced.mean()) if len(priced) else 0.0

    return JsonResponse({
        "total_rows": total_rows,
        "unique_suppliers": unique_suppliers,
        "unique_cities": unique_cities,
        "unique_states": unique_states,
        "median_price": round(median_price, 2),
        "avg_price": round(avg_price, 2),
    })


def api_top_cities(request):
    df = load_df()
    df = apply_filters(df, request.GET.get("state"), request.GET.get("keyword"))

    if df.empty or "city" not in df.columns:
        return JsonResponse({"labels": [], "values": []})

    s = df["city"].value_counts().head(12)
    return JsonResponse({"labels": s.index.tolist(), "values": s.values.tolist()})


def api_top_states(request):
    df = load_df()
    df = apply_filters(df, request.GET.get("state"), request.GET.get("keyword"))

    if df.empty or "state" not in df.columns:
        return JsonResponse({"labels": [], "values": []})

    s = df["state"].value_counts().head(12)
    return JsonResponse({"labels": s.index.tolist(), "values": s.values.tolist()})


def api_price_buckets(request):
    df = load_df()
    df = apply_filters(df, request.GET.get("state"), request.GET.get("keyword"))

    if df.empty or "price_bucket" not in df.columns:
        return JsonResponse({"labels": [], "values": []})

    s = df["price_bucket"].value_counts()

    labels = []
    values = []
    # prefer common buckets first if present
    for p in ["Low (<10k)", "Mid (10k-50k)", "High (50k+)", "Unknown"]:
        if p in s.index:
            labels.append(p)
            values.append(int(s[p]))

    for k, v in s.items():
        if k not in labels:
            labels.append(k)
            values.append(int(v))

    return JsonResponse({"labels": labels[:8], "values": values[:8]})


def api_price_hist(request):
    df = load_df()
    df = apply_filters(df, request.GET.get("state"), request.GET.get("keyword"))

    if df.empty or "price_numeric" not in df.columns:
        return JsonResponse({"bins": [], "counts": []})

    x = pd.to_numeric(df["price_numeric"], errors="coerce").dropna()
    if x.empty:
        return JsonResponse({"bins": [], "counts": []})

    # clip extreme tail so histogram looks nice
    p99 = float(x.quantile(0.99))
    x = x[x <= p99]

    counts, edges = np.histogram(x.values, bins=12)
    bins = [f"{int(edges[i]):,}-{int(edges[i+1]):,}" for i in range(len(edges) - 1)]
    return JsonResponse({"bins": bins, "counts": counts.tolist()})


def api_scatter_rating_price(request):
    df = load_df()
    df = apply_filters(df, request.GET.get("state"), request.GET.get("keyword"))

    if df.empty or "price_numeric" not in df.columns or "rating" not in df.columns:
        return JsonResponse({"points": []})

    d = df.dropna(subset=["price_numeric", "rating"]).copy()
    if d.empty:
        return JsonResponse({"points": []})

    p99 = d["price_numeric"].quantile(0.99)
    d = d[d["price_numeric"] <= p99]

    if len(d) > 1500:
        d = d.sample(1500, random_state=42)

    points = [{"x": float(r), "y": float(p)} for r, p in zip(d["rating"], d["price_numeric"])]
    return JsonResponse({"points": points})


# ✅ IMPORTANT: This was missing (your URLs require it)
def api_mini_rows(request):
    """
    Returns small sample rows for overview mini table.
    Fixes empty cells by returning None for missing numerics and "" for missing strings.
    """
    df = load_df()
    df = apply_filters(df, request.GET.get("state"), request.GET.get("keyword"))

    n = request.GET.get("n", "8")
    try:
        n = max(1, min(50, int(n)))
    except Exception:
        n = 8

    if df.empty:
        return JsonResponse({"rows": []})

    cols = [c for c in ["product_name", "supplier_name", "city", "price_numeric"] if c in df.columns]
    view = df[cols].head(n)

    rows = []
    for _, row in view.iterrows():
        rows.append({
            "product_name": safe_text(row.get("product_name")),
            "supplier_name": safe_text(row.get("supplier_name")),
            "city": safe_text(row.get("city")),
            "price_numeric": None if pd.isna(row.get("price_numeric")) else float(row.get("price_numeric")),
        })

    return JsonResponse({"rows": rows})
