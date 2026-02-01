"""
etl.py
============================================================
Professional-grade ETL pipeline for IndiaMART scraped data.

Run:
    python etl.py --input indiamart_21_keywords_products.csv

Disable SQLite:
    python etl.py --no-sqlite

Change output name:
    python etl.py --output clean_data.csv
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd


# ============================================================
# CONFIG
# ============================================================
@dataclass
class ETLConfig:
    input_file: str
    output_csv: str = "clean_data.csv"
    profile_report: str = "data_profile_report.csv"
    quality_issues: str = "data_quality_issues.csv"

    export_sqlite: bool = True
    output_db: str = "products.db"

    keep_phone_hash: bool = True
    winsorize_price: bool = True

    # ✅ Final output columns (ONLY what you want)
    final_columns: Tuple[str, ...] = (
        "search_keyword",
        "product_name",
        "supplier_name",
        "city",
        "state",
        "supplier_region",
        "rating",
        "price_numeric",
        "price_unit",
        "price_bucket",
        "product_url",
        "supplier_url",
        "catid",
        "mcatid",
        "itemid",
        "dispid",
        "scraped_at",
    )


# ============================================================
# LOGGING
# ============================================================
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("etl")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


# ============================================================
# HELPERS
# ============================================================
def to_snake_case(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name.lower()


def clean_text(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    if s.lower() in {"nan", "none", "null", ""}:
        return None
    return s


def parse_price_to_number(price_raw: Optional[str]) -> Optional[float]:
    s = clean_text(price_raw)
    if not s:
        return None
    low = s.lower()
    if "ask price" in low or "get quote" in low:
        return None
    s = s.replace("₹", "").replace(",", "")
    m = re.search(r"\d+(\.\d+)?", s)
    return float(m.group(0)) if m else None


def extract_price_unit(price_raw: Optional[str]) -> Optional[str]:
    s = clean_text(price_raw)
    if not s:
        return None
    m = re.search(r"/\s*([A-Za-z]+)", s)
    return m.group(1).strip().title() if m else None


def is_valid_url(u: Optional[str]) -> bool:
    s = clean_text(u)
    if not s:
        return False
    try:
        p = urlparse(s)
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False


def clean_phone_digits(phone_raw: Optional[str]) -> Optional[str]:
    s = clean_text(phone_raw)
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    return digits if digits else None


def sha256_hash(value: Optional[str], salt: str = "indiamart_etl_salt") -> Optional[str]:
    if not value:
        return None
    msg = f"{salt}::{value}"
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()


def normalize_keyword(k: Optional[str]) -> Optional[str]:
    s = clean_text(k)
    if not s:
        return None
    s = re.sub(r"\s+", " ", s.lower())
    fixes = {
        "bakery oven,": "bakery oven",
        "wet & dry vacuum cleaner": "wet and dry vacuum cleaner",
        "built-in dishwasher": "built in dishwasher",
        "semi automatic washing machine": "semi-automatic washing machine",
    }
    return fixes.get(s, s)


def normalize_city_state(x: Optional[str]) -> Optional[str]:
    s = clean_text(x)
    if not s:
        return None
    s = s.title()
    s = s.replace("Tamilnadu", "Tamil Nadu")
    return s


def supplier_region_from_state(state: Optional[str]) -> str:
    s = clean_text(state)
    if not s:
        return "Unknown"
    s = s.lower()

    south = {"tamil nadu", "kerala", "karnataka", "andhra pradesh", "telangana", "puducherry"}
    west = {"maharashtra", "gujarat", "goa", "rajasthan"}
    north = {"delhi", "punjab", "haryana", "uttar pradesh", "uttarakhand", "himachal pradesh", "jammu and kashmir"}
    east = {"west bengal", "odisha", "bihar", "jharkhand", "assam"}

    if s in south:
        return "South"
    if s in west:
        return "West"
    if s in north:
        return "North"
    if s in east:
        return "East"
    return "Other/Unknown"


def price_bucket(price: Optional[float]) -> str:
    if price is None or pd.isna(price):
        return "Unknown"
    if price < 10000:
        return "Low (<10k)"
    if price < 50000:
        return "Mid (10k-50k)"
    return "High (50k+)"


def iqr_bounds(series: pd.Series) -> Tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return (q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def winsorize_series(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if len(s) < 10:
        return series
    low, high = iqr_bounds(s)
    return series.clip(lower=low, upper=high)


# ============================================================
# PIPELINE STEPS
# ============================================================
def load_raw_data(config: ETLConfig, logger: logging.Logger) -> pd.DataFrame:
    if not os.path.exists(config.input_file):
        raise FileNotFoundError(f"Input file not found: {config.input_file}")

    df = pd.read_csv(config.input_file)
    logger.info("Loaded rows: %s", len(df))
    df.columns = [to_snake_case(c) for c in df.columns]

    # --------------------------------------------------------
    # Traceability fix:
    # Some older raw CSV exports may not include a `scraped_at`
    # column even though downstream expects it.
    # If missing (or entirely empty), backfill with the raw file
    # modified time (UTC ISO-8601) so the pipeline stays consistent.
    # --------------------------------------------------------
    if "scraped_at" not in df.columns or df.get("scraped_at").isna().all():
        try:
            from datetime import datetime, timezone

            ts = os.path.getmtime(config.input_file)
            inferred = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except Exception:
            # Last-resort fallback
            inferred = ""
        df["scraped_at"] = inferred

    return df


def standardize_fields(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(clean_text)

    if "search_keyword" in df.columns:
        df["search_keyword"] = df["search_keyword"].apply(normalize_keyword)

    if "city" in df.columns:
        df["city"] = df["city"].apply(normalize_city_state)

    if "state" in df.columns:
        df["state"] = df["state"].apply(normalize_city_state)

    if "supplier_name" in df.columns:
        df["supplier_name"] = df["supplier_name"].apply(lambda v: v.title() if isinstance(v, str) else v)

    return df


def fix_types_and_features(df: pd.DataFrame, config: ETLConfig) -> pd.DataFrame:
    # Price parsing
    if "price" in df.columns:
        df["price_numeric"] = df["price"].apply(parse_price_to_number)
        df["price_unit"] = df["price"].apply(extract_price_unit)
    else:
        df["price_numeric"] = pd.NA
        df["price_unit"] = pd.NA

    # Rating (NaN stays NaN)
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # IDs
    for col in ["catid", "mcatid", "itemid", "dispid"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Phone anonymization (optional)
    if config.keep_phone_hash and "phone" in df.columns:
        digits = df["phone"].apply(clean_phone_digits)
        df["phone_hash"] = digits.apply(lambda v: sha256_hash(v) if v else None)
    if "phone" in df.columns:
        df.drop(columns=["phone"], inplace=True)

    # Outliers (winsorize)
    if config.winsorize_price and pd.to_numeric(df["price_numeric"], errors="coerce").notna().sum() >= 10:
        df["price_numeric_winsor"] = winsorize_series(pd.to_numeric(df["price_numeric"], errors="coerce"))
    else:
        df["price_numeric_winsor"] = pd.to_numeric(df["price_numeric"], errors="coerce")

    # Derived features
    df["price_bucket"] = df["price_numeric_winsor"].apply(price_bucket)
    df["supplier_region"] = df.get("state", pd.Series([None] * len(df))).apply(supplier_region_from_state)

    return df


def validate_and_collect_issues(df: pd.DataFrame) -> pd.DataFrame:
    issues: List[Dict[str, str]] = []

    def add_issue(idx: int, issue: str) -> None:
        issues.append({"row_index": int(idx), "issue": issue})

    for idx, row in df.iterrows():
        if not row.get("product_name"):
            add_issue(idx, "missing_product_name")
        if not row.get("supplier_name"):
            add_issue(idx, "missing_supplier_name")

        pu = row.get("product_url")
        if pu and not is_valid_url(pu):
            add_issue(idx, "invalid_product_url")

        su = row.get("supplier_url")
        if su and not is_valid_url(su):
            add_issue(idx, "invalid_supplier_url")

        p = row.get("price_numeric")
        try:
            if p is not None and pd.notna(p) and float(p) <= 0:
                add_issue(idx, "non_positive_price")
        except Exception:
            pass

        r = row.get("rating")
        if r is not None and pd.notna(r) and (r < 0 or r > 5):
            add_issue(idx, "rating_out_of_range")

    return pd.DataFrame(issues, columns=["row_index","issue"])


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Fill ONLY city/state. Keep rating/price_numeric/price_unit as NaN.
    if "city" in df.columns:
        df["city"] = df["city"].fillna("Unknown")
    if "state" in df.columns:
        df["state"] = df["state"].fillna("Unknown")
    return df


def deduplicate(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    before = len(df)
    keys = []
    if "product_url" in df.columns:
        keys.append("product_url")
    if "dispid" in df.columns:
        keys.append("dispid")

    df = df.drop_duplicates(subset=keys, keep="first") if keys else df.drop_duplicates(keep="first")
    logger.info("Removed duplicates: %s", before - len(df))
    return df


def drop_critical_missing(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    before = len(df)
    for col in [c for c in ["product_name", "supplier_name"] if c in df.columns]:
        df = df[df[col].notna()]
    logger.info("Dropped rows missing required fields: %s", before - len(df))
    return df


def build_profile_report(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for col in df.columns:
        non_null = df[col].notna().sum()
        nulls = n - non_null
        uniq = df[col].nunique(dropna=True)
        dtype = str(df[col].dtype)
        sample = df[col].dropna().astype(str).head(3).tolist()
        rows.append(
            {
                "column": col,
                "dtype": dtype,
                "rows": int(n),
                "non_null": int(non_null),
                "nulls": int(nulls),
                "null_pct": round((nulls / n) * 100, 2) if n else 0,
                "unique": int(uniq),
                "sample_values": " | ".join(sample),
            }
        )
    return pd.DataFrame(rows).sort_values(by="null_pct", ascending=False)


def curate_final_dataset(df: pd.DataFrame, config: ETLConfig, logger: logging.Logger) -> pd.DataFrame:
    # Guarantee exact column set + order
    for col in config.final_columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[list(config.final_columns)].copy()


def export_outputs(
    df_final: pd.DataFrame,
    profile_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    config: ETLConfig,
    logger: logging.Logger,
) -> None:
    # ✅ Write NaN as literal "NaN" in the CSV (so you can SEE missing values)
    df_final.to_csv(config.output_csv, index=False, encoding="utf-8-sig", na_rep="NaN")
    profile_df.to_csv(config.profile_report, index=False, encoding="utf-8-sig")
    issues_df.to_csv(config.quality_issues, index=False, encoding="utf-8-sig")

    logger.info("Saved cleaned dataset: %s (rows=%s)", config.output_csv, len(df_final))
    logger.info("Saved profiling report: %s", config.profile_report)
    logger.info("Saved quality issues: %s (rows=%s)", config.quality_issues, len(issues_df))

    # ✅ Optional SQLite export (Bonus)
    if config.export_sqlite:
        try:
            conn = sqlite3.connect(config.output_db)
            df_final.to_sql("products", conn, if_exists="replace", index=False)
            conn.close()
            logger.info("Saved SQLite DB: %s (table=products)", config.output_db)
        except Exception as e:
            logger.error("SQLite export failed: %s", e)


# ============================================================
# CLI ARGS
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETL for IndiaMART scraped dataset")
    parser.add_argument(
        "--input",
        default="indiamart_21_keywords_products.csv",
        help="Path to raw scraped CSV file",
    )
    parser.add_argument("--no-sqlite", action="store_true", help="Disable SQLite export")
    parser.add_argument("--output", default="clean_data.csv", help="Output cleaned CSV name")
    return parser.parse_args()


def run_pipeline(config: ETLConfig, logger: logging.Logger) -> None:
    logger.info("Starting ETL pipeline...")

    df = load_raw_data(config, logger)
    df = standardize_fields(df)
    df = fix_types_and_features(df, config)

    issues_df = validate_and_collect_issues(df)

    df = handle_missing_values(df)
    df = deduplicate(df, logger)
    df = drop_critical_missing(df, logger)

    df_final = curate_final_dataset(df, config, logger)

    # Profile the FINAL curated dataset (so columns match clean_data.csv)
    profile_df = build_profile_report(df_final)

    export_outputs(df_final, profile_df, issues_df, config, logger)

    logger.info("ETL pipeline completed successfully.")


def main() -> None:
    logger = setup_logger()
    args = parse_args()

    config = ETLConfig(
        input_file=args.input,
        output_csv=args.output,
        export_sqlite=not args.no_sqlite,
    )

    run_pipeline(config, logger)


if __name__ == "__main__":
    main()

