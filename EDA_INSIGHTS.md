# EDA Insights & Hypotheses (IndiaMART sample)

Dataset: `clean_data.csv`  
Rows: **391** | Keywords (categories): **21**

## Key observations
- **Regional concentration:** Suppliers are dominated by the **South** region (~88.0%), with smaller shares from North/West and a tiny unknown share.
- **Top states by listings:** Tamil Nadu (265), Karnataka (50), Kerala (28), Delhi (16), Maharashtra (10).
- **Top cities by listings:** Coimbatore (195), Bengaluru (48), Chennai (16), New Delhi (16), Tiruppur (16).
- **Price bucket mix:** Mid (10k-50k) (106), Unknown (105), Low (<10k) (101), High (50k+) (79).
- **Missingness:** `price_numeric` missing in **26.9%** of rows; `rating` missing in **27.1%** of rows.

## Price distribution
- Median price (non-null): **25250**
- IQR (non-null): **52231**  
  Hypothesis: prices are **right-skewed** (many mid/low items with fewer high-value outliers), so median/IQR are more stable than mean.

## Rating vs price
- Correlation (rows with both rating and price): **-0.037**  
  Interpretation: in this sample, **rating and price are weakly related**; quality perception may depend more on brand/service than price.

## Data quality gaps (expected for marketplace scraping)
- Some listings do not show price/rating publicly → leads to “Unknown” price bucket and missing numeric price.
- Location fields may be incomplete or inconsistent (city/state missing for a subset).
- Product naming is noisy (brands/specs mixed into titles) — tokenization in `reports/top_product_tokens.csv` can help normalize categories later.
