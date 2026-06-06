# RetentionIQ — Churn Prediction with Causal Uplift Scoring

A churn model tells you *who will leave*. This tells you *who is worth saving*.

Most retention systems predict churn probability and call the top N customers. The problem: some of those customers will churn regardless of what you do, and some will stay anyway. Both are wasted calls.

RetentionIQ adds a causal uplift layer on top of churn prediction to identify the **Winnable** segment — customers who will churn without intervention but will stay if contacted. This lets a retention team invest their call budget where it actually changes outcomes.

---

## Problem Statement

A telecom company runs a monthly retention campaign with a fixed call budget. Given a list of customers, the goal is to answer two questions:

1. Who is likely to churn?
2. Of those, who will actually respond to an intervention?

Answering only question 1 is table stakes. Answering both is what this project does.

---

## Approach

**Stage 1 — Churn Model (T0)**
An XGBoost classifier trained on the IBM Telco dataset predicts the probability of churn for each customer in the absence of any intervention. Threshold is optimized using F2-score to minimize false negatives — missing a churner costs more than a wasted call.

**Stage 2 — Intervention Model (T1)**
A second XGBoost model is trained on the treatment group only, learning the probability of churn *even after* a call. This is the T-Learner component of the uplift architecture.

**Uplift Score**
```
uplift = P(churn | no call) - P(churn | called)
       = T0.predict(X)      - T1.predict(X)
```

Clipped at 0 — a call cannot increase churn probability. High uplift means the intervention meaningfully reduces churn risk for that customer.

**Call Prioritization**
Customers are ranked by uplift score. The `should_call` flag is assigned to the top N% based on the team's call budget, making the model directly parameterizable by business capacity.

---

## Results

| Metric | Value |
|---|---|
| Churn model AUC-ROC | 0.8404 |
| Recall (at optimized threshold) | 0.9225 |
| Precision (at optimized threshold) | 0.4238 |
| F2 Score | optimized over 101 thresholds (0.00 to 1.00) |
| CV AUC (5-fold) | 0.8455 ± 0.0049 |
| Churn rate — flagged for call (top 20%) | 48.0% |
| Churn rate — not flagged (remaining 80%) | 21.2% |

The uplift model concentrates actual churners into the intervention pool — 48% churn rate among flagged customers vs 21% in the rest, using only 20% of the customer base. A naive churn-ranked call list does not achieve this concentration because it includes lost causes — high-churn customers the intervention cannot save.

---

## Dataset

IBM Telco Customer Churn dataset — 7,032 customers, 20 raw features, 26% churn rate.

Source: [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Feature Engineering

Key decisions made during EDA and correlation analysis:

- **Dropped** `gender` — identical churn rate across male/female, zero discriminative power
- **Dropped** all `_No internet service` dummy columns — structurally identical to `InternetService_No` by dataset logic (a customer without internet cannot have Yes/No for any add-on service)
- **Dropped** engineered aggregates (`service_count`, `has_addon`, `is_monthly`, `is_autopay`, `senior_no_support`, `avg_monthly_spend`, tenure buckets) after finding high multicollinearity (>0.5) with their source columns — individual features carry more specific signal than the summaries
- **Retained** 21 features after correlation filtering (threshold: |r| > 0.05 with Churn) and multicollinearity analysis

Final feature matrix: 7,032 rows × 21 features, all numeric, 0 nulls.

---

## Uplift Modeling — Honest Framing

The IBM Telco dataset does not include experiment data — there is no record of which customers were contacted in a past campaign. The treatment column is simulated via randomized 50/50 assignment to replicate an A/B test structure, with a 35% save rate applied to treated churners to simulate intervention effect.

The balance of features across treatment and control groups was verified before model training — all feature means were within 3% across groups, confirming the randomization was unbiased.

In production, the simulated treatment would be replaced with actual A/B test logs from a real retention campaign. The architecture — T-Learner with budget-parameterized uplift ranking — remains unchanged.

---

## Tech Stack

- Python, Pandas, NumPy
- XGBoost
- scikit-learn (StratifiedKFold, threshold optimization, evaluation metrics)
- MLflow (experiment tracking, model logging)
- Matplotlib, Seaborn

---

## Project Structure

```
retentioniq/
├── churn-intervention.ipynb    # full notebook: EDA, feature engineering, modeling, uplift
└── README.md
```

---

## How to Run

Open the notebook on Kaggle and attach the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset. Run all cells in order. MLflow experiment logs are written to the Kaggle working directory.

---

## What's Next (Future Work)

- Replace simulated treatment with real A/B test outcome data
- Build a Streamlit app: upload raw customer CSV, get churn probability + uplift score + `should_call` flag per customer, downloadable as a prioritized call list
- Add Optuna hyperparameter tuning for both models
- Dockerize the app for deployment
