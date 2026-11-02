
"""main.py
One-command runner for the fresh IndiaMART data pipeline.

Run the complete pipeline:
    python main.py

Optional:
    python main.py --stage scrape
    python main.py --stage etl
    python main.py --stage eda
    python main.py --headless
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd=str(ROOT))

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run the fresh IndiaMART data pipeline."
    )

    p.add_argument(
        "--stage",
        choices=["all", "scrape", "etl", "eda"],
        default="all"
    )

    p.add_argument(
        "--headless",
        action="store_true",
        help="Run Chrome without displaying the browser."
    )

    p.add_argument("--max-pages", type=int, default=1)
    p.add_argument("--max-cards", type=int, default=80)

    p.add_argument("--driver", default=None)

    p.add_argument("--scraped-csv", default=None)
    p.add_argument("--clean-csv", default="fresh_clean_data.csv")

    p.add_argument("--checkpoint", default=None)
    p.add_argument("--partial", default=None)

    args = p.parse_args()

    # Generate a unique ID for this run.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use unique scraped/checkpoint/partial filenames.
    scraped_csv = args.scraped_csv or (
        f"scraped_{run_id}.csv"
        if args.stage in ("all", "scrape")
        else "indiamart_21_keywords_products.csv"
    )

    checkpoint = args.checkpoint or f"checkpoint_{run_id}.json"
    partial = args.partial or f"partial_{run_id}.csv"

    print("\n======================================")
    print("  INDIA MART DATA PIPELINE")
    print("======================================")
    print(f"Run ID: {run_id}")
    print(f"Scraped data: {scraped_csv}")
    print(f"Cleaned data: {args.clean_csv}")

    # Stage 1: Fresh scraping
    if args.stage in ("all", "scrape"):
        print("\n[STAGE 1] SCRAPING FRESH DATA")

        cmd = [
            sys.executable,
            "scraper.py",
            "--max-pages", str(args.max_pages),
            "--max-cards", str(args.max_cards),
            "--out", scraped_csv,
            "--checkpoint", checkpoint,
            "--partial", partial,
        ]

        if args.driver:
            cmd.extend(["--driver", args.driver])

        if args.headless:
            cmd.append("--headless")

        run(cmd)

    # Stage 2: ETL cleaning
    if args.stage in ("all", "etl"):
        print("\n[STAGE 2] CLEANING DATA")

        run([
            sys.executable,
            "etl.py",
            "--input", scraped_csv,
            "--output", args.clean_csv,
        ])

    # Stage 3: Data analysis
    if args.stage in ("all", "eda"):
        print("\n[STAGE 3] GENERATING ANALYTICS")

        run([sys.executable, "analysis.py"])

    print("\n======================================")
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("======================================")


if __name__ == "__main__":
    main()
