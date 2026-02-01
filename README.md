# B2B-Data-Pipeline-IndiaMART

End-to-end B2B data engineering pipeline for the **Slooze take-home challenge**:
**IndiaMART scraping → ETL → EDA (10 charts) → Django dashboard**

---

## ✅ What this project does

### Part A — Data Collection (IndiaMART)
- Scrapes product listings for multiple keywords/categories
- Includes retry + backoff and polite rate-limiting to reduce blocking
- Outputs structured data as CSV (and optional DB artifacts)

### Part B — Exploratory Data Analysis (EDA)
- Cleans and standardizes scraped data (price parsing, buckets, regions)
- Generates **exactly 10 professional charts** saved in `/plots`
- Produces summary reports in `/reports` and insights in `EDA_INSIGHTS.md`

### Dashboard (Bonus)
- Django dashboard to explore KPIs, charts and data table

---

## 📁 Project Structure

- `scraper.py` — IndiaMART data collector
- `etl.py` — cleaning + transformation + schema enforcement
- `analysis.py` — EDA + chart generation (10 plots)
- `main.py` — runs pipeline end-to-end
- `clean_data.csv` — cleaned dataset output
- `plots/` — generated chart images
- `reports/` — summary CSV reports
- `dashboard/` — Django app (local dashboard)

---

## ⚙️ Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt


▶️ Run (Quick)
Run full pipeline (ETL + EDA)
python main.py

Run EDA only (generates plots + reports)
python analysis.py

Run Django dashboard
cd dashboard
python manage.py runserver


Then open:
http://127.0.0.1:8000/

📊 Outputs
Charts (exactly 10)

Saved in /plots:

KPI Cards

Line Chart

Bar Chart

Donut Chart

Histogram (Price distribution)

Map approximation (City index)

Combo chart (Bar + Line)

Treemap

Waterfall

Scatter plot

Reports

Saved in /reports:

dataset summary stats

missingness & quality checks

outlier listing

top product keywords/tokens

🧾 Notes on Data Quality

Some listings do not display price or rating publicly → appears as missing values

Outliers exist due to bulk/industrial listings → handled via reporting + bucketing

Contact

For the Slooze submission: careers@slooze.xyz


This README hits **everything recruiters asked**: run instructions, outputs, structure, quality notes.

---

# 3) Add screenshots to README (makes it 2x more impressive)
In GitHub, you can add:

```md
## Dashboard Preview
![Dashboard](plots/01_kpi_cards.png)


And/or upload your dashboard screenshot images into a folder like /screenshots/ and reference them.

