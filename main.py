"""main.py
One-command runner for the Slooze take-home pipeline.

Examples:
  python main.py --headless --max-pages 1 --max-cards 80
  python main.py --stage scrape
  python main.py --stage etl
  python main.py --stage eda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["all", "scrape", "etl", "eda"], default="all")
    p.add_argument("--headless", action="store_true", help="Run Chrome in headless mode (scraper).")
    p.add_argument("--max-pages", type=int, default=1, help="Max pages per keyword (scraper).")
    p.add_argument("--max-cards", type=int, default=80, help="Max product cards per keyword (scraper).")
    p.add_argument("--driver", default=None, help="Path to chromedriver.exe (optional; if omitted Selenium Manager will try to auto-manage).")
    p.add_argument("--scraped-csv", default="indiamart_21_keywords_products.csv")
    p.add_argument("--clean-csv", default="clean_data.csv")
    args = p.parse_args()

    if args.stage in ("all", "scrape"):
        cmd = [
            sys.executable, "scraper.py",
            "--max-pages", str(args.max_pages),
            "--max-cards", str(args.max_cards),
            "--out", args.scraped_csv,
        ]
        if args.driver:
            cmd.extend(["--driver", args.driver])
        if args.headless:
            cmd.append("--headless")
        run(cmd)

    if args.stage in ("all", "etl"):
        run([
            sys.executable, "etl.py",
            "--input", args.scraped_csv,
            "--output", args.clean_csv,
        ])

    if args.stage in ("all", "eda"):
        run([sys.executable, "analysis.py"])


if __name__ == "__main__":
    main()
