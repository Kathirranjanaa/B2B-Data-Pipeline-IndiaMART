
"""
scraper.py
============================================================
Professional-grade IndiaMART search crawler using Selenium.

Features
--------
1. Supports both Template7 and MLP product card layouts.
2. Explicit waits using WebDriverWait.
3. Retry with backoff on page-loading failures.
4. Defensive parsing using safe_text and safe_attr.
5. Manual IndiaMART phone-number and OTP login.
6. Show More clicks and controlled scrolling.
7. Best-effort pagination.
8. Rate limiting and random delays.
9. Checkpointing and resume support.
10. Incremental CSV saving.
11. Stable output schema compatible with the ETL pipeline.
12. URL validation and duplicate removal.

Run examples
------------
python scraper.py
python scraper.py --headless
python scraper.py --out indiamart_raw.csv
python scraper.py --keywords keywords.txt --max-pages 2 --max-cards 60
python scraper.py --jsonl indiamart_raw.jsonl
"""

from __future__ import annotations

import os
import argparse
import json
import logging
import random
import time
import urllib.parse

from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

from selenium import webdriver
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
    StaleElementReferenceException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


# ============================================================
# CONFIG
# ============================================================

DEFAULT_KEYWORDS = [
    "led tv",
    "android tv",
    "smart led tv",
    "dry iron",
    "travel iron",
    "electric iron",
    "Fully Automatic Washing Machine",
    "Semi-Automatic Washing Machine",
    "Laundry Washing Machine",
    "Electric Oven",
    "Kitchen Oven",
    "Bakery Oven",
    "Countertop Dishwasher",
    "Portable Dishwasher",
    "Built In Dishwasher",
    "Split AC",
    "Window AC",
    "Cassette AC",
    "Wet And Dry vacuum cleaner",
    "Handheld vacuum cleaner",
    "Robot vacuum cleaner",
]


@dataclass
class CrawlConfig:

    driver_path: Optional[str] = None

    base_url: str = (
        "https://dir.indiamart.com/search.mp?ss={keyword}"
    )

    out_csv: str = "indiamart_21_keywords_products.csv"
    out_jsonl: Optional[str] = None

    # Resume support
    checkpoint_file: str = "crawl_checkpoint.json"
    partial_csv: str = "crawl_partial.csv"

    flush_every_keywords: int = 1

    headless: bool = False

    page_load_timeout: int = 60
    wait_timeout: int = 20

    # Retry configuration
    max_retries: int = 3
    retry_backoff_base: float = 4.0

    # Delay between keywords
    min_delay: float = 4.0
    max_delay: float = 8.0

    random_jitter: Tuple[float, float] = (0.5, 1.5)

    # Optional rate limit
    max_rpm: Optional[int] = None

    # Optional user agent
    user_agent: Optional[str] = None

    # Show More and scrolling
    click_show_more: bool = True
    max_show_more_clicks: int = 2

    scroll_after_clicks: bool = True
    scroll_steps: int = 3

    # Crawl limits
    max_pages: int = 1
    max_cards: Optional[int] = None

    # Stable output schema
    schema: Tuple[str, ...] = (
        "search_keyword",
        "product_name",
        "product_url",
        "supplier_name",
        "supplier_url",
        "price",
        "phone",
        "city",
        "state",
        "locality",
        "location_ui",
        "rating",
        "image",
        "catid",
        "mcatid",
        "itemid",
        "dispid",
        "brand",
        "capacity",
        "power",
        "ac_type",
        "function_type",
        "isq_attributes",
        "scraped_at",
    )


# ============================================================
# LOGGING
# ============================================================

def setup_logger() -> logging.Logger:

    logger = logging.getLogger("crawler")
    logger.setLevel(logging.INFO)

    if not logger.handlers:

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ============================================================
# LOW-LEVEL HELPERS
# ============================================================

def safe_text(parent, css: str) -> str:
    """Safely extract text from a child element."""

    try:
        return parent.find_element(
            By.CSS_SELECTOR, css
        ).text.strip()

    except Exception:
        return ""


def safe_attr(parent, css: str, attr: str) -> str:
    """Safely extract an attribute from a child element."""

    try:
        value = parent.find_element(
            By.CSS_SELECTOR, css
        ).get_attribute(attr)

        return (value or "").strip()

    except Exception:
        return ""


def attr_or_empty(element, attr: str) -> str:
    """Safely extract an attribute from the element itself."""

    try:
        return (element.get_attribute(attr) or "").strip()

    except Exception:
        return ""


def is_valid_url(url: str) -> bool:
    """Validate HTTP and HTTPS URLs."""

    if not url:
        return False

    if url.strip().lower().startswith("javascript:"):
        return False

    try:
        parsed = urlparse(url)

        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.netloc)
        )

    except Exception:
        return False


def sleep_jitter(
    base_seconds: float,
    jitter_range: Tuple[float, float] = (0.5, 1.5),
) -> None:
    """Sleep for a randomized duration."""

    time.sleep(
        base_seconds * random.uniform(
            jitter_range[0],
            jitter_range[1],
        )
    )


class RateLimiter:
    """Limit page navigations per minute."""

    def __init__(self, max_rpm: Optional[int]) -> None:

        self.max_rpm = max_rpm

        self._min_interval = (
            60.0 / max_rpm
            if max_rpm and max_rpm > 0
            else 0.0
        )

        self._last_ts = 0.0

    def wait(self) -> None:

        if self._min_interval <= 0:
            return

        now = time.time()
        target = self._last_ts + self._min_interval

        if now < target:
            time.sleep(target - now)

        self._last_ts = time.time()


# ============================================================
# INDIA MART SPECIFIC PARSERS
# ============================================================

def extract_price(card) -> str:
    """Extract price from a Template7 product card."""

    price = safe_text(
        card,
        "span.template7-product-price",
    )

    if price:
        return price

    return safe_text(card, "p.getquote")


def extract_phone(card) -> str:
    """Extract phone number when it is visible in the card."""

    phone = safe_text(card, "span.pns_h")

    if phone:
        return phone

    phone = safe_text(
        card,
        "p.contactnumber span.pns_h",
    )

    return phone or ""


def decode_isq(isq_raw: str) -> Dict[str, str]:
    """
    Decode URL-encoded product attributes.

    Example:
    Brand%3ADaikin%23Capacity%3A2.8%20kW
    """

    if not isq_raw:
        return {}

    decoded = urllib.parse.unquote(isq_raw)

    result: Dict[str, str] = {}

    for part in decoded.split("#"):

        if ":" in part:

            key, value = part.split(":", 1)

            key = key.strip()
            value = value.strip()

            if key:
                result[key] = value

    return result


def parse_card(
    card,
    keyword: str,
) -> Optional[Dict[str, str]]:
    """
    Parse a Template7 product card.

    Returns a dictionary containing the stable output schema.
    """

    # Product information
    product_name = safe_text(
        card,
        "a.template7-product-name",
    )

    product_url = safe_attr(
        card,
        "a.template7-product-name",
        "href",
    )

    # Supplier information
    supplier_name = safe_text(
        card,
        "a.template7-seller-name",
    )

    supplier_url = safe_attr(
        card,
        "a.template7-seller-name",
        "href",
    )

    # Critical fields
    if not product_name or not supplier_name:
        return None

    # Validate URLs
    if product_url and not is_valid_url(product_url):
        product_url = ""

    if supplier_url and not is_valid_url(supplier_url):
        supplier_url = ""

    # Price and phone
    price = extract_price(card)
    phone = extract_phone(card)

    # Location
    location_ui = safe_text(
        card,
        "span[itemprop='addressLocality']",
    )

    city = location_ui

    # Other information
    rating = attr_or_empty(card, "data-rating")
    state = attr_or_empty(card, "data-state")
    locality = attr_or_empty(card, "data-locality")

    # Product IDs
    catid = attr_or_empty(card, "data-catid")
    mcatid = attr_or_empty(card, "data-mcatid")
    itemid = attr_or_empty(card, "data-itemid")
    dispid = attr_or_empty(card, "data-dispid")

    # Product image
    img = safe_attr(
        card,
        "img.template7-product-image",
        "src",
    )

    if not img:
        img = attr_or_empty(card, "data-origimg")

    # Product attributes
    isq_raw = attr_or_empty(card, "data-isq")
    isq_map = decode_isq(isq_raw)

    brand = (
        isq_map.get("Brand", "")
        or isq_map.get("Brand Name", "")
    )

    capacity = (
        isq_map.get("Capacity", "")
        or isq_map.get("Capacity(Litre)", "")
    )

    power = isq_map.get("Power", "")

    function_type = (
        isq_map.get("Function", "")
        or isq_map.get("Function Type", "")
    )

    ac_type = isq_map.get("Type", "")

    return {
        "search_keyword": keyword,
        "product_name": product_name,
        "product_url": product_url,
        "supplier_name": supplier_name,
        "supplier_url": supplier_url,
        "price": price,
        "phone": phone,
        "city": city,
        "state": state,
        "locality": locality,
        "location_ui": location_ui,
        "rating": rating,
        "image": img,
        "catid": catid,
        "mcatid": mcatid,
        "itemid": itemid,
        "dispid": dispid,
        "brand": brand,
        "capacity": capacity,
        "power": power,
        "ac_type": ac_type,
        "function_type": function_type,
        "isq_attributes": "; ".join(
            [
                f"{key}={value}"
                for key, value in isq_map.items()
            ]
        ),
    }


def parse_mlp_card(
    card,
    keyword: str,
) -> Optional[Dict[str, str]]:
    """
    Parse a product from IndiaMART's newer MLP card layout.

    MLP cards use different CSS selectors from Template7 cards.
    Both parsers return the same output structure.
    """

    # Product information
    product_name = safe_text(
        card,
        "a.mlpCardTitle",
    )

    product_url = safe_attr(
        card,
        "a.mlpCardTitle",
        "href",
    )

    # Supplier information
    supplier_name = safe_text(
        card,
        "a.mlpCardSeller",
    )

    supplier_url = safe_attr(
        card,
        "a.mlpCardSeller",
        "href",
    )

    # Critical fields
    if not product_name or not supplier_name:
        return None

    # Validate URLs
    if product_url and not is_valid_url(product_url):
        product_url = ""

    if supplier_url and not is_valid_url(supplier_url):
        supplier_url = ""

    # Price
    price = safe_text(
        card,
        "p.mlpCardPrice",
    )

    # Location
    location_ui = safe_text(
        card,
        "span.mlpCardCity",
    )

    city = location_ui

    # Product image
    img = safe_attr(
        card,
        "img.mlpCardImage",
        "src",
    )

    # These fields are not currently exposed by the MLP card.
    # Leave them empty instead of inventing values.
    phone = ""
    state = ""
    locality = ""
    rating = ""
    catid = ""
    mcatid = ""
    itemid = ""
    dispid = ""
    brand = ""
    capacity = ""
    power = ""
    ac_type = ""
    function_type = ""
    isq_attributes = ""

    return {
        "search_keyword": keyword,
        "product_name": product_name,
        "product_url": product_url,
        "supplier_name": supplier_name,
        "supplier_url": supplier_url,
        "price": price,
        "phone": phone,
        "city": city,
        "state": state,
        "locality": locality,
        "location_ui": location_ui,
        "rating": rating,
        "image": img,
        "catid": catid,
        "mcatid": mcatid,
        "itemid": itemid,
        "dispid": dispid,
        "brand": brand,
        "capacity": capacity,
        "power": power,
        "ac_type": ac_type,
        "function_type": function_type,
        "isq_attributes": isq_attributes,
    }


# ============================================================
# PAGE INTERACTIONS
# ============================================================

def click_show_more(
    driver,
    wait: WebDriverWait,
    logger: logging.Logger,
    max_clicks: int,
) -> None:
    """Click the Show More button when available."""

    for i in range(max_clicks):

        try:

            button = wait.until(
                EC.element_to_be_clickable(
                    (By.CLASS_NAME, "showMoreBtn")
                )
            )

            driver.execute_script(
                "arguments[0].click();",
                button,
            )

            logger.info(
                "Clicked 'Show More' (%s/%s)",
                i + 1,
                max_clicks,
            )

            sleep_jitter(2.0)

        except Exception:
            return


def scroll_page(
    driver,
    steps: int = 3,
) -> None:
    """Scroll the page to trigger lazy loading."""

    for _ in range(steps):

        driver.execute_script(
            "window.scrollBy(0, document.body.scrollHeight / 2);"
        )

        sleep_jitter(1.5)


def wait_for_cards(wait: WebDriverWait) -> None:
    """Wait for either supported IndiaMART card layout."""

    wait.until(
        EC.presence_of_element_located(
            (
                By.CSS_SELECTOR,
                "div.template7-product-card, article.mlpCard",
            )
        )
    )


# ============================================================
# DRIVER SETUP
# ============================================================

def build_driver(
    config: CrawlConfig,
) -> webdriver.Chrome:

    options = Options()

    options.page_load_strategy = "eager"

    options.add_argument("--disable-gpu")
    options.add_argument("start-maximized")
    options.add_argument("--no-sandbox")

    default_ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    options.add_argument(
        f"--user-agent={config.user_agent or default_ua}"
    )

    if config.headless:
        options.add_argument("--headless=new")

    if config.driver_path:

        driver = webdriver.Chrome(
            service=Service(config.driver_path),
            options=options,
        )

    else:

        # Selenium Manager manages the ChromeDriver automatically.
        driver = webdriver.Chrome(
            options=options,
        )

    driver.set_page_load_timeout(
        config.page_load_timeout
    )

    return driver


# ============================================================
# CORE CRAWL LOGIC
# ============================================================

def build_search_url(
    config: CrawlConfig,
    keyword: str,
    page: int,
) -> str:
    """Build the IndiaMART search URL."""

    base = config.base_url.format(
        keyword=urllib.parse.quote(keyword)
    )

    if page <= 1:
        return base

    return f"{base}&pg={page}"


def load_with_retry(
    driver,
    wait: WebDriverWait,
    url: str,
    logger: logging.Logger,
    config: CrawlConfig,
    limiter: Optional[RateLimiter] = None,
) -> bool:
    """Load a search URL with retry and manual login support."""

    for attempt in range(
        1,
        config.max_retries + 1,
    ):

        try:

            if limiter:
                limiter.wait()

            driver.get(url)

            logger.info(
                "Page title: %s",
                driver.title,
            )

            # Manual login is requested only once per browser session.
            if not getattr(
                driver,
                "_indiamart_login_confirmed",
                False,
            ):

                input(
                    "\nComplete IndiaMART phone number and OTP login "
                    "in Chrome. After the login popup disappears "
                    "and product results are accessible, press Enter "
                    "here to continue..."
                )

                # Confirm that both login popups are no longer visible.
                wait.until(
                    lambda d: (
                        "Login to connect with suppliers"
                        not in d.find_element(
                            By.TAG_NAME,
                            "body",
                        ).text
                        and
                        "Login with One Time Password"
                        not in d.find_element(
                            By.TAG_NAME,
                            "body",
                        ).text
                    )
                )

                driver._indiamart_login_confirmed = True

                logger.info(
                    "IndiaMART login confirmed."
                )

            wait_for_cards(wait)

            logger.info(
                "Product cards successfully detected."
            )

            return True

        except TimeoutException as exc:

            logger.warning(
                "Timeout on attempt %s/%s: %s",
                attempt,
                config.max_retries,
                exc,
            )

            # Collect diagnostics to help investigate loading failures.
            try:

                page_info = driver.execute_script(
                    """
                    return {
                        readyState: document.readyState,
                        bodyLength: document.body.innerText.length,
                        bodyText: document.body.innerText.substring(0, 500),
                        divCount: document.querySelectorAll('div').length,
                        linkCount: document.querySelectorAll('a').length,
                        cardElements: Array.from(
                            document.querySelectorAll('[class*="card"]')
                        ).slice(0, 15).map(el => ({
                            tag: el.tagName,
                            className: el.className
                        }))
                    };
                    """
                )

                logger.info(
                    "PAGE DIAGNOSTICS: %s",
                    page_info,
                )

            except WebDriverException as check_error:

                logger.warning(
                    "Could not inspect page: %s",
                    check_error,
                )

        except WebDriverException as exc:

            logger.warning(
                "WebDriver error on attempt %s/%s: %s",
                attempt,
                config.max_retries,
                exc,
            )

        backoff = (
            config.retry_backoff_base * attempt
        )

        logger.warning(
            "Retrying after %.1f seconds...",
            backoff,
        )

        sleep_jitter(
            backoff,
            config.random_jitter,
        )

    return False


def scrape_keyword(
    driver,
    wait: WebDriverWait,
    keyword: str,
    logger: logging.Logger,
    config: CrawlConfig,
    limiter=None,
) -> List[Dict[str, str]]:
    """
    Scrape results for one keyword across multiple pages.

    Supports both Template7 and MLP card layouts.
    """

    all_rows: List[Dict[str, str]] = []

    cards_seen = 0
    parsed = 0
    skipped = 0
    pages_ok = 0

    for page in range(
        1,
        config.max_pages + 1,
    ):

        url = build_search_url(
            config,
            keyword,
            page,
        )

        logger.info(
            "Scraping keyword: %s (page %s/%s)",
            keyword,
            page,
            config.max_pages,
        )

        ok = load_with_retry(
            driver,
            wait,
            url,
            logger,
            config,
            limiter=limiter,
        )

        if not ok:

            logger.error(
                "Skipping keyword page (failed loads): %s page=%s",
                keyword,
                page,
            )

            continue

        pages_ok += 1

        # Expand results when the button is available.
        if config.click_show_more:

            click_show_more(
                driver,
                wait,
                logger,
                config.max_show_more_clicks,
            )

        # Scroll to trigger lazy loading.
        if config.scroll_after_clicks:

            scroll_page(
                driver,
                config.scroll_steps,
            )

        # ----------------------------------------------------
        # DETECT BOTH INDIA MART CARD LAYOUTS
        # ----------------------------------------------------

        template7_cards = driver.find_elements(
            By.CSS_SELECTOR,
            "div.template7-product-card",
        )

        mlp_cards = driver.find_elements(
            By.CSS_SELECTOR,
            "article.mlpCard",
        )

        logger.info(
            "Cards found: Template7=%s | MLP=%s",
            len(template7_cards),
            len(mlp_cards),
        )

        # Pair each card with its correct parser.
        cards = (
            [
                (card, parse_card)
                for card in template7_cards
            ]
            +
            [
                (card, parse_mlp_card)
                for card in mlp_cards
            ]
        )

        # ----------------------------------------------------
        # PARSE PRODUCT CARDS
        # ----------------------------------------------------

        for card, parser_func in cards:

            cards_seen += 1

            # Apply max_cards across all pages for this keyword.
            if (
                config.max_cards is not None
                and parsed >= config.max_cards
            ):
                break

            try:

                row = parser_func(
                    card,
                    keyword,
                )

                if row:

                    all_rows.append(row)
                    parsed += 1

                else:

                    skipped += 1

            except StaleElementReferenceException:

                skipped += 1
                continue

            except Exception:

                skipped += 1
                continue

        # Stop pagination when the card limit is reached.
        if (
            config.max_cards is not None
            and parsed >= config.max_cards
        ):
            break

        # Delay between pages.
        if page < config.max_pages:

            sleep_jitter(
                2.0,
                config.random_jitter,
            )

    # --------------------------------------------------------
    # KEYWORD SUMMARY
    # --------------------------------------------------------

    logger.info(
        "Keyword summary: '%s' | pages_ok=%s "
        "cards_seen=%s parsed=%s skipped=%s",
        keyword,
        pages_ok,
        cards_seen,
        parsed,
        skipped,
    )

    return all_rows


# ============================================================
# DATAFRAME AND OUTPUT HELPERS
# ============================================================

def enforce_schema(
    rows: List[Dict[str, str]],
    schema: Tuple[str, ...],
) -> pd.DataFrame:
    """Create a DataFrame with a stable column order."""

    df = pd.DataFrame(rows)

    for column in schema:

        if column not in df.columns:
            df[column] = ""

    return df[list(schema)]


def write_jsonl(
    rows: List[Dict[str, str]],
    path: str,
) -> None:
    """Write one JSON object per line."""

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        for row in rows:

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# CHECKPOINTING AND RESUME
# ============================================================

def load_checkpoint(path: str) -> dict:
    """Load the previous crawl checkpoint."""

    if not path or not os.path.exists(path):

        return {
            "completed_keywords": [],
            "seen_keys": [],
        }

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if not isinstance(data, dict):

            return {
                "completed_keywords": [],
                "seen_keys": [],
            }

        data.setdefault(
            "completed_keywords",
            [],
        )

        data.setdefault(
            "seen_keys",
            [],
        )

        return data

    except Exception:

        return {
            "completed_keywords": [],
            "seen_keys": [],
        }


def save_checkpoint(
    path: str,
    completed_keywords: list[str],
    seen_keys: set[str],
) -> None:
    """Save crawl progress."""

    data = {
        "completed_keywords": completed_keywords,
        "seen_keys": list(seen_keys)[:200000],
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def append_rows_csv(
    path: str,
    rows: list[dict],
    schema: list[str],
) -> None:
    """Append scraped rows to the partial CSV."""

    if not rows:
        return

    df = enforce_schema(
        rows,
        tuple(schema),
    )

    write_header = not os.path.exists(path)

    df.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# MAIN CRAWL PROCESS
# ============================================================

def run_crawl(
    keywords: Iterable[str],
    config: CrawlConfig,
    logger: logging.Logger,
) -> None:

    driver = build_driver(config)

    wait = WebDriverWait(
        driver,
        config.wait_timeout,
    )

    limiter = RateLimiter(
        config.max_rpm,
    )

    # Load checkpoint.
    checkpoint = load_checkpoint(
        config.checkpoint_file,
    )

    completed = set(
        [
            keyword.strip().lower()
            for keyword in checkpoint.get(
                "completed_keywords",
                [],
            )
            if str(keyword).strip()
        ]
    )

    seen_keys: set[str] = set(
        [
            str(value)
            for value in checkpoint.get(
                "seen_keys",
                [],
            )
        ]
    )

    keywords_seen = 0

    processed_keywords: list[str] = list(
        checkpoint.get(
            "completed_keywords",
            [],
        )
    )

    try:

        for keyword in keywords:

            keyword = keyword.strip()

            if not keyword:
                continue

            keyword_key = keyword.lower()

            # Resume support.
            if keyword_key in completed:

                logger.info(
                    "[RESUME] Skipping already completed keyword: %s",
                    keyword,
                )

                continue

            keywords_seen += 1

            rows = scrape_keyword(
                driver,
                wait,
                keyword,
                logger,
                config,
                limiter=limiter,
            )

            # Add timestamp to the scraped batch.
            scraped_at = datetime.now(
                timezone.utc
            ).isoformat()

            output_rows: list[dict] = []

            # Incremental duplicate removal.
            for row in rows:

                row["scraped_at"] = scraped_at

                key = (
                    str(row.get("product_url", "")).strip()
                    + "|"
                    + str(row.get("dispid", "")).strip()
                )

                if key == "|":

                    key = (
                        str(row.get("product_name", ""))
                        + "|"
                        + str(row.get("supplier_name", ""))
                        + "|"
                        + keyword_key
                    ).strip()

                if key in seen_keys:
                    continue

                seen_keys.add(key)

                output_rows.append(row)

            # Save progress after each keyword.
            append_rows_csv(
                config.partial_csv,
                output_rows,
                list(config.schema),
            )

            # Update checkpoint.
            completed.add(keyword_key)

            processed_keywords.append(keyword)

            save_checkpoint(
                config.checkpoint_file,
                processed_keywords,
                seen_keys,
            )

            # Delay between keywords.
            sleep_seconds = random.uniform(
                config.min_delay,
                config.max_delay,
            )

            sleep_jitter(
                sleep_seconds,
                config.random_jitter,
            )

    finally:

        driver.quit()

    # --------------------------------------------------------
    # BUILD FINAL CSV FROM PARTIAL CSV
    # --------------------------------------------------------

    if not os.path.exists(config.partial_csv):

        logger.warning(
            "No partial CSV found: %s",
            config.partial_csv,
        )

        df = enforce_schema(
            [],
            config.schema,
        )

    else:

        df = pd.read_csv(
            config.partial_csv,
            encoding="utf-8-sig",
        )

    # Remove duplicates.
    dedupe_keys = [
        key
        for key in [
            "product_url",
            "dispid",
        ]
        if key in df.columns
    ]

    before = len(df)

    if dedupe_keys:

        df = df.drop_duplicates(
            subset=dedupe_keys,
            keep="first",
        )

    else:

        df = df.drop_duplicates(
            keep="first",
        )

    logger.info(
        "Removed duplicates: %s",
        before - len(df),
    )

    # Save final CSV.
    df.to_csv(
        config.out_csv,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "Saved CSV: %s (rows=%s)",
        config.out_csv,
        len(df),
    )

    # Optional JSONL output.
    if config.out_jsonl:

        write_jsonl(
            df.to_dict(orient="records"),
            config.out_jsonl,
        )

        logger.info(
            "Saved JSONL: %s",
            config.out_jsonl,
        )

    # Final summary.
    logger.info(
        "Crawl finished. keywords=%s total_rows=%s unique_rows=%s",
        keywords_seen,
        before,
        len(df),
    )


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def read_keywords_from_file(
    path: str,
) -> List[str]:
    """Read one keyword per line."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="IndiaMART crawler (professional)"
    )

    parser.add_argument(
        "--driver",
        default=None,
        help="Path to chromedriver.exe (optional)",
    )

    parser.add_argument(
        "--out",
        default="indiamart_21_keywords_products.csv",
        help="Output CSV filename",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint JSON filename",
    )

    parser.add_argument(
        "--partial",
        default=None,
        help="Partial CSV filename",
    )

    parser.add_argument(
        "--jsonl",
        default=None,
        help="Optional JSONL output filename",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chrome in headless mode",
    )

    parser.add_argument(
        "--keywords",
        default=None,
        help="Path to keywords.txt (one keyword per line)",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum pages per keyword",
    )

    parser.add_argument(
        "--max-cards",
        type=int,
        default=None,
        help="Maximum cards per keyword",
    )

    parser.add_argument(
        "--show-more-clicks",
        type=int,
        default=2,
        help="Number of Show More clicks",
    )

    parser.add_argument(
        "--min-delay",
        type=float,
        default=4.0,
        help="Minimum delay between keywords",
    )

    parser.add_argument(
        "--max-delay",
        type=float,
        default=8.0,
        help="Maximum delay between keywords",
    )

    parser.add_argument(
        "--max-rpm",
        type=int,
        default=None,
        help="Maximum page navigations per minute",
    )

    parser.add_argument(
        "--user-agent",
        type=str,
        default=None,
        help="Optional custom User-Agent",
    )

    return parser.parse_args()


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main() -> None:

    logger = setup_logger()

    args = parse_args()

    # Load keywords.
    if args.keywords:

        keywords = read_keywords_from_file(
            args.keywords,
        )

    else:

        keywords = DEFAULT_KEYWORDS

    # Build configuration.
    config = CrawlConfig(

        driver_path=args.driver or None,

        out_csv=args.out,

        out_jsonl=args.jsonl,

        checkpoint_file=(
            args.checkpoint
            or f"{args.out}.checkpoint.json"
        ),

        partial_csv=(
            args.partial
            or f"{args.out}.partial.csv"
        ),

        headless=args.headless,

        max_pages=max(
            1,
            int(args.max_pages),
        ),

        max_cards=args.max_cards,

        max_show_more_clicks=args.show_more_clicks,

        min_delay=float(args.min_delay),

        max_delay=float(args.max_delay),

        max_rpm=args.max_rpm,

        user_agent=args.user_agent,
    )

    # Start the crawler.
    run_crawl(
        keywords,
        config,
        logger,
    )


if __name__ == "__main__":
    main()
