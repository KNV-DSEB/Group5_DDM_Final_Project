# Group5_DDM_Final_Project — Silent Exit: E-Commerce Customer Churn

**Data-Driven Marketing — Semester 6 Final Project**

An end-to-end churn analytics pipeline for a non-contractual e-commerce platform, where
customers leave silently rather than cancelling a subscription ("silent exit"). The project
covers exploratory analysis, economic cost modelling, machine-learning prediction (LR + RF +
SHAP), risk segmentation, and a Priority Save Zone for targeted retention.

---

## Overview

**Problem.** Non-contractual churn is defined by recency: a customer who stops purchasing
without notification. The notebook engineers a churn label from days-since-last-purchase and
validates it against platform transaction history.

**Pipeline.**
1. Data loading & label validation (Module 0–2)
2. Economic impact — revenue at risk, CLV 3-year, retention ROI (Module 3)
3. Behavioural EDA — order trends, category signals, device & session patterns (Modules 4–5)
4. Feature engineering — 26 RFM + behavioural + demographic features (Module 6)
5. Modelling — Logistic Regression, Random Forest, SHAP, 5-fold CV (Module 7)
6. Risk segmentation — probability tiers, Priority Save Zone, retention budget (Module 8)
7. Category & product strategy — H4 hypothesis tests (Module 9)
8. Acquisition channel & dead-spend analysis (Module 10)
9. Summary dashboard + scored customer export (Module 11)

---

## Repository Structure

```text
Group5_DDM_Final_Project/
├── README.md                          This file
├── requirements.txt                   Python dependencies
├── .gitignore
├── Group5_DDM_Final_Notebook.ipynb    Main notebook (end-to-end, run top-to-bottom)
├── data/
│   ├── customers.csv                  8,000 customers with demographics & churn label
│   ├── orders.csv                     25,000 transactions (2020-01-01 to 2026-03-30)
│   ├── product_summary.csv            140 product-level aggregates
│   └── monthly_revenue.csv            75 months of platform KPIs
├── report/
│   ├── Phase2_IEEE_Paper.tex          Main IEEE paper (LaTeX source)
│   ├── Phase0_Academic_Foundation.md  Literature review & hypothesis framework
└── output/
    ├── customers_scored.csv           All 8,000 customers with churn probability & risk tier
    ├── ecommerce_churn_dashboard_p2.png  Summary dashboard figure
    └── fig_m*.png                     Per-module publication figures (pre-generated)
```
---

## How to Run

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook Group5_DDM_Final_Notebook.ipynb
```

Run all cells **top-to-bottom**. The notebook is self-contained: it reads data from `data/`,
trains all models in memory, and saves figures to the working directory.
Pre-generated figures are available in `output/` without re-running.

---

## Report

| File | Description |
|---|---|
| `report/Phase2_IEEE_Paper.tex` | Main IEEE-format paper (compile with pdflatex or Overleaf) |
| `report/Phase0_Academic_Foundation.md` | Research questions, hypotheses, and literature survey |

To compile the paper locally:
```bash
cd report
pdflatex Phase2_IEEE_Paper.tex
```
Or upload `Phase2_IEEE_Paper.tex` to [Overleaf](https://overleaf.com) for one-click compilation.

---

## Data

All four CSV files are included in `data/`. The notebook reads them via:

```python
DATA_DIR = 'data'
customers = pd.read_csv(os.path.join(DATA_DIR, 'customers.csv'))
```

| File | Rows | Key columns |
|---|---|---|
| `customers.csv` | 8,000 | `customer_id`, `membership_tier`, `churned`, `CLV`-related fields |
| `orders.csv` | 25,000 | `order_id`, `category`, `total_amount_usd`, `device_used`, `returned` |
| `product_summary.csv` | 140 | `category`, `return_rate`, `avg_discount_pct`, `avg_delivery_days` |
| `monthly_revenue.csv` | 75 | `revenue_usd`, `unique_customers`, `new_customers`, `avg_order_value` |

---

## Key Results (computed at runtime)

| Metric | Value |
|---|---|
| Overall churn rate | 8.94% (715 / 8,000) |
| RF ROC-AUC (hold-out) | 0.748 |
| RF 5-fold CV ROC-AUC | 0.723 ± 0.023 |
| Behavioral lift over demographics | +0.233 ROC-AUC |
| Revenue at risk (churned %) | 7.0% of total platform revenue |
| H4 (return rate → churn) | r = −0.571, p = 0.033 — **rejected** |
