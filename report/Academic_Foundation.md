# PHASE 0 — ACADEMIC FOUNDATION
## Silent Exit — Using Behavioral Data to Predict and Price E-Commerce Customer Churn (2020–2026)

---

## STEP 1 — THEORETICAL BACKGROUND

### 1. Churn in Non-Contractual E-Commerce

In contractual settings (telecom, SaaS, banking products with explicit term commitments), churn is observable and dated: when a customer cancels, both the fact and the timing are recorded by the firm's systems. **E-commerce is non-contractual.** Customers do not announce that they have left. They simply stop placing orders. Churn is therefore a *latent* state that, in principle, must be inferred from behavioral evidence.

In practice, mature e-commerce platforms address this by maintaining an internal `churned` flag inside their CRM, derived from a business-rule definition (for example, prolonged inactivity together with absence of any active subscription, account closure, or explicit opt-out). The empirical setting of this study follows that pattern: the dataset arrives with a pre-existing operational churn label. The role of the analyst is therefore not to *engineer* a churn label from scratch, but to **cross-validate the supplied label against behavioral evidence** before using it for modeling. Concretely, the pipeline computes a recency-based pseudo-label using the 75th percentile of `days_since_last_purchase` as a cutoff and compares it to the supplied label via a confusion matrix. High agreement confirms that the operational label is recency-consistent; controlled disagreement indicates that the label encodes additional signals beyond recency — which is itself a useful diagnostic, since a label encoding richer information than a single threshold is preferable for downstream prediction.

The economic stakes remain unchanged regardless of how the label is sourced. Reichheld (2001) documented that retaining an existing customer costs five to seven times less than acquiring a new one, and that small retention improvements compound into large profit gains. The silent nature of e-commerce churn makes the problem more dangerous than in contractual settings: by the time a manager notices that a customer has stopped buying, the customer may already be transacting elsewhere. This justifies the *silent exit* framing of the present study.

### 2. Customer Value and Pricing Churn

Not all churners are economically equal. A customer with a 3-year CLV in the low double digits and a customer with a 3-year CLV in the high triple digits represent very different losses to the firm, yet a probability-only churn model treats them identically. **The economic question is therefore not "who will churn?" but "how much value walks out the door when each customer churns?"**

Two complementary metrics operationalize this:

- **Revenue at risk:** the historical spend already attributable to customers currently flagged as churned. This is realized economic damage already on the books — the dollar value of relationships that have decayed below the retention threshold.
- **CLV at risk:** the projected forward value of customers currently still classified as active but predicted by the model to be at high risk of churn. This is prospective damage if no intervention is taken. In this study, the per-customer CLV is approximated by a 3-year horizon (`CLV_3yr`), computed from average order value and annualized purchase frequency.

Within a fixed retention budget, value-weighted prioritization (concentrating spending on high-CLV at-risk customers) outperforms probability-only prioritization (treating all high-probability churners equally), because the expected return on intervention scales with both the conversion uplift *and* the value of the customer being saved. This logic underpins the **Priority Save Zone (PSZ)** construct used later in the pipeline: the intersection of high predicted churn probability with high membership tier (Platinum or Gold), which by construction contains the customers where per-capita retention spending has the largest expected return.

### 3. Behavioral Drivers of Churn (RFM and Beyond)

The Recency–Frequency–Monetary (RFM) heuristic, formalized by Hughes (1994) and validated econometrically by Bult and Wansbeek (1995), remains the most robust low-dimensional summary of customer behavior. *Recency* — `days_since_last_purchase` — is the single strongest churn predictor across virtually every empirical study in non-contractual retail, because it is the most direct observable signal of disengagement.

Pure RFM, however, is descriptive: it ranks customers but does not capture the *dynamics* of behavior. Modern churn models extend RFM with derived features that describe how a customer's behavior has changed over time and across the catalog. The pipeline implements three layers of behavioral predictors beyond raw RFM:

- **Recency-weighted velocity:** `orders_last_90d` and `orders_last_180d` capture the *rate* of recent purchasing in addition to the lifetime count, distinguishing a long-tenured customer who recently went quiet from a long-tenured customer who is still active.
- **Catalog interaction:** `category_diversity` (number of distinct categories purchased) captures breadth of engagement. Customers who buy across more categories typically churn less, because they have more reasons to return.
- **On-platform engagement:** `session_duration_avg`, `pages_viewed_avg`, `pct_mobile`, `wishlist_items`, and `reviews_given` tap pre-purchase behavior on the platform itself, capturing intent shifts that may *precede* an actual stop in transactions. Where recency is the lagging indicator of churn, engagement metrics function as leading indicators.
- **Customer-level satisfaction:** `pct_returned`, `avg_review_score`, and `avg_order_rating` summarize how the customer has experienced past purchases. Persistently high returns or low ratings signal expectation–reality mismatch at the individual level.

These behavioral features capture interactions and non-linearities that demographic features (age, gender, country, tier) cannot, and they map directly to actions the marketing team can take: re-engagement campaigns, category cross-sells, dormant-customer outreach, or platform-experience interventions when engagement metrics drop.

### 4. Product and Service Quality Signals

Product-level metrics provide a second layer of explanation that operates orthogonally to customer-level behavior:

- **Return rate** by category reflects mismatch between expectation (set by product page, photos, reviews) and reality.
- **Average rating** captures aggregate product satisfaction.
- **Average delivery days** captures fulfillment performance.
- **Average discount percentage** is ambiguous: high discounts can drive acquisition but may also signal a *discount trap* — customers who only buy on promotion and churn when promotions stop.

A customer whose favorite category exhibits high return rates and long delivery times is exposed to two structural friction sources independent of their own loyalty. The pipeline captures this by joining each customer's preferred category against `product_summary.csv` and propagating the relevant product-quality features into the customer feature matrix. At the platform level, the same signals are aggregated to reveal whether categories with elevated return rates or discount intensity coincide with elevated category-level churn — the empirical test of the discount-trap hypothesis.

### 5. Macro–Micro Linkage

Platform-level monthly metrics — revenue, average discount, return rate, ratio of new to returning customers — often move together with individual churn pressure:

- A platform-wide rise in **discount intensity** that is not matched by a rise in revenue suggests the platform is buying transactions rather than earning loyalty — a leading indicator of future churn.
- A rising **return rate** signals broad product-experience deterioration.
- A falling **returning-customer ratio** signals that the customer base is becoming structurally more fragile.

These macro indicators function as early-warning signals: they shift weeks or months before individual-level churn metrics do, because individual recency requires waiting for the absence of a transaction. Integrating macro signals (from `monthly_revenue.csv`) with micro-level prediction creates a complete monitoring stack, ranging from platform-health KPI dashboards down to customer-level risk scores. The pipeline implements both layers so that macro warnings and micro predictions can be read against each other.

### 6. Explainable ML for Churn

For a marketing audience, model accuracy is necessary but not sufficient. The marketing team must understand *why* a customer is at risk in order to design the right intervention, and the analytics team must be able to defend model decisions to risk and compliance.

**Logistic Regression** (LR) provides linear, signed coefficients that translate directly into "this feature increases churn odds by *X*%." It is the natural baseline.

**Random Forest** (RF) and other tree-based ensembles capture non-linear effects and feature interactions that LR cannot — for instance, the qualitatively different behavior of a high-spend customer who suddenly stops buying versus a low-spend customer who never bought much in the first place.

**SHAP** (Shapley Additive Explanations; Lundberg & Lee, 2017) provides theoretically grounded local and global explanations for tree-based models. SHAP enables both the global feature ranking (which features dominate the model) and the per-customer narrative ("this specific customer is at high risk because their recency is high *and* their order regularity has dropped"), which is what the marketing team actually consumes. Crucially, SHAP allows direct comparison of the relative weight of behavioral versus demographic features in the trained model — the empirical operationalization of H2.

The combination — LR for transparency, RF for accuracy, SHAP for interpretability — is the pragmatic stack for explainable churn analytics in non-contractual e-commerce. In this study, the stack is applied to a real labeled dataset, so model outputs are directly interpretable in dollar terms and operational terms.

---

## STEP 2 — LITERATURE REVIEW

### Theme 1 — ML Models for Churn Prediction (Telecom → Retail → E-commerce)

The methodological evolution of churn prediction follows a consistent arc across industries: from Logistic Regression and decision trees in the 1990s, to Random Forest and SVM in the 2000s, to gradient-boosted trees in the late 2010s. Verbeke et al.\ (2012), benchmarking on telecom data, established the *profit-driven* evaluation framework — comparing classifiers on expected campaign profit rather than pure accuracy. Ahmad, Jafar, and Aljoumaa (2019) compared LR, RF, gradient boosting, and XGBoost on nine million telecom records and found gradient boosting achieving the highest AUC. Recent benchmarking by Grinsztajn, Oyallon, and Varoquaux (2022) confirmed that on tabular data with fewer than one million rows, gradient-boosted and tree-based models consistently outperform deep learning architectures. **Limitation common to this literature:** most studies evaluate models on accuracy metrics in isolation, without grounding the comparison in an economic objective.

### Theme 2 — Behavioral Features versus Demographics

A consistent finding across churn studies is that *behavior beats demographics* by a large margin. Demographic features (age, gender, country, even tenure) produce mediocre AUCs in isolation; adding behavioral features (recency, frequency, monetary, order regularity, category breadth, engagement) typically lifts AUC substantially. The intuition is that demographics describe *who the customer is* (slow-changing, weakly predictive of action), while behavior describes *what the customer is doing* (fast-changing, directly predictive of next action). De Caigny, Coussement, and De Bock (2018) demonstrated this gap on retail data and proposed hybrid Logistic-Regression-Tree models to combine the interpretability of LR with the non-linearity of trees. **Limitation:** the literature documents *that* behavioral features outperform but rarely *quantifies the marginal lift* on a clean demographics-only versus full-feature comparison on the same dataset and same train/test split.

### Theme 3 — Customer Value, Product Quality, and Platform Health

Reichheld's (2001) economic argument for retention has been formalized in the customer-equity literature (Gupta and Lehmann, 2005), which treats the customer base as a financial asset whose value is the sum of individual CLVs. Fader, Hardie, and Lee (2005a, 2005b) provided the probabilistic CLV machinery (BG/NBD + Gamma-Gamma) for non-contractual settings, and Ascarza (2018) delivered the critical refinement that targeting customers with the highest churn probability does *not* necessarily maximize retention ROI — because some are "lost causes" and others are "sure things." The economically optimal target is the *persuadable* high-value customer, and within a fixed budget, value-weighted prioritization dominates probability-only prioritization.

In parallel, the marketing literature documents two opposing roles of discount intensity: discounts drive trial and acquisition, but heavy reliance creates a discount trap in which customers anchor on promotional pricing and churn when promotions stop. Return rates have a similarly dual reading: returns enable trial through generous policies but also signal expectation–reality mismatch. A smaller but growing strand of literature treats platform-level monthly metrics (returning-customer ratio, discount-vs-revenue trend, return-rate trend) as leading indicators of customer-base health. **Limitation across this theme:** while the theoretical literature is clear, empirical studies rarely close the loop by reporting concrete revenue-at-risk and CLV-at-risk figures alongside model accuracy, and rigorous correlation analysis between category-level discount/return and customer-level churn on production data is rare.

---

## STEP 3 — RESEARCH GAP

The three literature streams above converge on a coherent gap. The technical literature on ML churn prediction is mature: multiple studies have established that tree-based models outperform linear baselines and that behavioral features outperform demographics. The economic literature on customer value and CLV is similarly mature: Reichheld, Gupta, Fader, and Ascarza have all provided robust frameworks for valuing customers and for prioritizing retention. The marketing literature on product quality, discount strategy, and platform-level indicators has produced rich qualitative insight.

What has *not* been done — and what this project addresses — is the *integration* of these three layers in a single, end-to-end, data-driven pipeline applied to a real labeled e-commerce dataset. Most empirical churn studies stop at AUC. They report a model and its accuracy but do not quantify (i) how many dollars of revenue or CLV are at risk, (ii) how concentrated that risk is across customer segments, (iii) whether high-discount or high-return categories systematically generate higher churn, (iv) how platform-level macro signals relate to micro-level churn pressure, or (v) which specific customers the marketing team should target tomorrow under a binding budget constraint.

**This project addresses that gap by building an integrated, data-driven churn pipeline applied to a real labeled e-commerce dataset that (a) uses rich behavioral features grounded in RFM theory and extended with recency-weighted velocity, category breadth, on-platform engagement, and customer-level satisfaction signals; (b) prices churn in dollar terms — historical revenue at risk and forward 3-year CLV at risk; (c) interprets results in light of product-level (return rate, discount, rating, delivery) and macro-level (monthly revenue, discount intensity, returning-customer ratio) signals; and (d) translates predictions into a tiered, value-weighted retention budget framework with a Priority Save Zone of named customer segments ready for marketing action.**

---

## STEP 4 — RESEARCH QUESTIONS AND HYPOTHESES

### Research Questions

**RQ1 — Predictive drivers.** Which behavioral and value-based features are most strongly associated with churn, and how does their predictive importance compare to demographic features when measured by both Random Forest Gini importance and SHAP global mean absolute value?

**RQ2 — Concentration of churn.** How is churn distributed across membership tiers (Platinum, Gold, Silver, Free), age groups, countries, gender, and customer-preferred product categories? Is the distribution monotonic with loyalty, or does it exhibit non-trivial structure?

**RQ3 — Economic stakes.** How much historical revenue is attributable to churned customers, and how does forward 3-year CLV at risk distribute across membership tiers? Is the dollar-denominated risk concentrated in the high-tier minority or diffused across the lower-tier majority?

**RQ4 — Retention prioritization.** Which customers and segments should be prioritized for retention spending under realistic budget constraints? How should retention budget be allocated across the cells of a risk-tier × membership-tier matrix, and what does the resulting Priority Save Zone look like in operational terms?

### Hypotheses

**H1 — Concentration.** Churn is concentrated in specific membership tiers and demographic / behavioral segments rather than being uniformly distributed across the customer base. The hypothesis is tested by computing per-segment churn rates (by tier, country, age group, gender, and preferred category) and assessing whether the variation across segments is meaningfully larger than statistical noise. *Theoretical basis:* customer-equity literature (Gupta and Lehmann, 2005) and Pareto patterns in retail customer-revenue distribution.

**H2 — Behavioral lift over demographics.** Behavioral features (recency, recency-weighted velocity, total spend, CLV proxy, order rating, engagement metrics) outperform demographic features (age, gender, country, tier, acquisition channel) in predicting churn. The hypothesis is operationalized along two complementary axes implemented in the pipeline:

1. *Performance comparison:* training a Random Forest on the demographics-only subset and a Random Forest on the full feature set with the same train/test split, then comparing ROC-AUC and PR-AUC.
2. *Importance ranking:* computing SHAP global importance on the full Random Forest and verifying that behavioral features dominate the top of the ranking, while demographic features cluster at the bottom.

*Theoretical basis:* the behavioral-features-beat-demographics finding consistently observed across the churn ML literature (De Caigny et al., 2018; Ahmad et al., 2019).

**H3 — Disproportionate per-capita value loss.** High-tier customers (Platinum, Gold), despite being a numerical minority, carry a disproportionate share of CLV per capita relative to lower-tier customers. The hypothesis is tested by computing mean `CLV_3yr` by membership tier, comparing the per-capita value gradient across Platinum, Gold, Silver, and Free, and verifying that the Priority Save Zone — defined as the intersection of High-risk customers with Platinum or Gold tier — exhibits substantially higher mean CLV than the platform average. *Theoretical basis:* Reichheld (2001), Gupta and Lehmann (2005), and the Ascarza (2018) value-weighted targeting principle.

**H4 — Discount and return correlation with category churn.** At the product-category level, higher average discount percentage and higher average return rate are positively correlated with churn rate among customers whose favorite category falls in that bucket. The hypothesis is tested by computing Pearson correlations between category-level `avg_discount_pct`, `return_rate` (from `product_summary.csv`) and the category-level customer churn rate, with both significance ($p$-value) and direction ($r$-sign) reported. *Theoretical basis:* the discount-trap and product-mismatch arguments from the marketing-strategy literature.

---

## STEP 5 — CONTRIBUTION STATEMENT

**Academic contribution.** This project integrates three literatures that have evolved in parallel — ML churn prediction, customer-value economics, and product/macro marketing analytics — into a single end-to-end pipeline applied to a real labeled e-commerce dataset rather than a synthetic toy. The empirical contribution is the explicit, side-by-side measurement of the marginal predictive lift of behavioral features over demographics on the same Random Forest specification, the explicit dollar-denomination of churn (revenue at risk, forward CLV at risk, retention scenario ROI), and the explicit translation of predicted probabilities into a value-weighted retention budget matrix. The combination is rare in the published churn literature, where economic translation is typically left as an exercise for the reader. The project also operationalizes the Ascarza (2018) critique by replacing flat probability-targeting with risk-tier × membership-tier prioritization that explicitly weights by customer value. Methodologically, the use of a pre-existing operational churn label that is *cross-validated* against behavioral recency — rather than engineered directly from recency — provides a cleaner separation between the label being predicted and the features used to predict it.

**Managerial contribution.** The deliverables include a reproducible Python notebook, a tiered retention budget framework with named action recommendations per (risk-tier × membership-tier) cell, and a *Priority Save Zone* — the intersection of high churn probability and high membership tier that should receive the largest per-capita budget. The framework is designed to be directly consumable by an e-commerce marketing or CRM team: rather than receiving a list of probabilities, the team receives a list of named customers, the dollar value at stake for each, the maximum economic budget per intervention, and a recommended action template. This converts a typical churn model from an analytics artifact into an operational targeting system, addressing the persistent gap between churn-model accuracy reports and managerial impact (Rößler and Schoder, 2022). The pipeline additionally surfaces category-level discount and return diagnostics together with macro KPIs (monthly revenue, discount intensity, return rate, returning-customer ratio), giving the marketing team a coherent dashboard that links individual-level risk scores upward to platform-level health indicators.

---

## STEP 6 — LIMITATIONS

The framework's limitations are operational and inferential.

**Single-platform scope.** All empirical results are derived from one platform's customer base. Generalization to other e-commerce platforms — with different customer demographics, product mixes, fulfillment networks, and competitive contexts — is not guaranteed. Replication across platforms would strengthen external validity.

**Operational label noise.** The `churned` flag is the platform's business-rule definition of departure, which may not perfectly align with academic notions of latent churn. The label may include customers who paused rather than left, exclude customers who quietly disengaged but remained on the books, or reflect business-specific operational rules (account-status flags, opt-out events) that other platforms would define differently. The pipeline mitigates this by validating the label against a recency-based pseudo-label, but residual noise cannot be fully eliminated.

**Observational, non-causal correlations.** All relationships reported in the pipeline — the discount/return associations (H4), the SHAP-based driver rankings, and the tier-churn pattern — establish *associations*, not causal effects. A category that exhibits high returns and high churn does not necessarily cause that churn through the return mechanism; both may be driven by an unobserved factor such as product-market fit. Causal claims would require either randomized experimentation (A/B tests) or quasi-experimental identification (instrumental variables, difference-in-differences) that are out of scope for the present pipeline.

**Static snapshot rather than real-time scoring.** The models are trained on a fixed point-in-time customer state. In production, customer behavior evolves continuously, and concept drift will erode model performance over time. Deployment would require periodic retraining, drift monitoring, rolling re-validation of the churn label against updated recency distributions, and ideally a temporal train/test split (training on orders before a cutoff date, testing on behavior after) rather than the random stratified split used here for benchmarking.

**No real intervention experiment.** The retention budget framework rests on assumed conversion-uplift parameters drawn from industry benchmarks rather than from a randomized field experiment on this platform's customers. The reported lift therefore represents an expected-value calculation under reasonable assumptions, not a measured causal effect of intervention. Bridging this gap is the natural next phase of the project.

---

*End of Phase 0*
