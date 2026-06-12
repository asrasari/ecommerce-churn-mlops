# 5-Minute Presentation Script — E-Commerce Churn MLOps with MLflow

**Author:** Asra Sarı (2101640) · AIN-3009
**Setup before you start:** MLflow server running (`mlflow server --host 127.0.0.1 --port 8080
--backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts`), browser open at
**http://127.0.0.1:8080**, and a terminal ready. Have the GitHub repo tab open too.

> Timing target: ~5 minutes. Each section below has a time budget, what to SAY, and what to DO.

---

## 0:00–0:30 — Intro & problem (slide / talk, no demo)

**Say:**
> "My project is an end-to-end MLOps system for predicting **e-commerce customer churn** —
> whether a customer will stop buying. Churn matters because keeping a customer is far cheaper
> than acquiring a new one. But the focus of this project isn't just the model — it's using
> **MLflow** to manage the whole machine-learning lifecycle: tracking experiments, tuning,
> a model registry, deployment, and monitoring. The dataset has ~3,900 customers, 10 features,
> and a churn label that's about 17% positive."

---

## 0:30–1:30 — Experiment tracking (DEMO in MLflow UI)

**Say:**
> "First, **experiment tracking**. I trained three models — Logistic Regression, Random Forest,
> and Gradient Boosting — and logged each one to MLflow as a 'run'. For every run MLflow records
> the parameters, the metrics, plots, and the model itself."

**Do:**
1. Click the **`ecommerce-churn`** experiment in the left sidebar.
2. Point at the run table — "each row is one model I trained."
3. Tick the checkboxes next to **logistic_regression**, **random_forest**, **gradient_boosting**
   → click **Compare**.
4. Point at the metrics: "Random Forest is the best — ROC-AUC **0.966** vs 0.88 for logistic
   regression."

**Say (concept):**
> "The point is reproducibility — I can always see exactly what produced each result."

---

## 1:30–2:15 — Hyperparameter tuning with Optuna (DEMO)

**Say:**
> "Next I tuned the model with **Optuna**. It ran 20 experiments automatically, each trying
> different settings — and every single trial is logged to MLflow as its own run."

**Do:**
1. Back in `ecommerce-churn`, find the **`optuna-tuning`** run.
2. Click the small arrow/toggle to expand its **nested runs** (the 20 trials).
3. "Each of these is one trial. The best reached AUC ~0.958."

**Say:**
> "So MLflow gives me a full audit trail of the search, not just the final answer."

---

## 2:15–3:00 — Model Registry: Staging → Production (DEMO)

**Say:**
> "Once I have the best model, I promote it through the **Model Registry** — this is how teams
> manage which model is actually live."

**Do:**
1. Click **Models** in the top nav.
2. Open **`ecommerce-churn-model`** → show **Version 1** with stage **Production**.

**Say (concept):**
> "My code registers the best run, moves it to **Staging**, runs a quality check — it must beat
> an ROC-AUC threshold of 0.85 — and only then promotes it to **Production**. That gate is what
> stops a bad model from ever going live."

---

## 3:00–4:00 — Deployment / serving (DEMO in terminal)

**Say:**
> "The Production model is then served as a live service. I load it and send it a customer record
> over HTTP, and it returns a churn prediction in real time."

**Do (have this ready in a terminal; the Flask service `python -m src.serve_api` should already
be running on port 1234):**
```bash
curl -X POST http://127.0.0.1:1234/invocations -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["Tenure","WarehouseToHome","NumberOfDeviceRegistered","SatisfactionScore","NumberOfAddress","Complain","DaySinceLastOrder","CashbackAmount","PreferedOrderCat","MaritalStatus"], "data": [[1,29,4,5,9,1,0,120.5,"Mobile Phone","Single"]]}}'
```
**Say:** "This high-risk customer — short tenure, filed a complaint — returns **1**, predicted to churn."

> If short on time, skip the live curl and just say this sentence — the concept is what matters.

---

## 4:00–4:40 — Monitoring & drift (DEMO)

**Say:**
> "Finally, **monitoring**. In production, data changes over time — that's called drift. I simulate
> incoming batches and, for each one, log live metrics back to MLflow and compute a drift score
> called **PSI**. When PSI crosses a threshold, the feature is flagged as drifted — which in real
> life would trigger a retrain."

**Do:**
1. Open the **`ecommerce-churn-monitoring`** experiment.
2. Show the 5 `batch_` runs; click the **Chart** view and point at `psi_CashbackAmount` rising
   across batches.

---

## 4:40–5:00 — Orchestration + close

**Say:**
> "All of this is wired into an **Airflow** pipeline so it can run on a schedule automatically,
> and the whole project is on GitHub. In short: MLflow let me manage the complete lifecycle —
> track, tune, register, deploy, and monitor — in one reproducible system. Thank you!"

---

## Quick-fire Q&A prep

- **"Why Random Forest?"** — It had the best ROC-AUC (0.966) and is robust on tabular data with
  little tuning. AUC matters here because the classes are imbalanced (17% churn).
- **"What is ROC-AUC?"** — A score from 0.5 (random) to 1.0 (perfect) for how well the model ranks
  a churner above a non-churner. Better than accuracy when classes are imbalanced.
- **"What is PSI?"** — Population Stability Index: measures how much a feature's distribution has
  shifted vs. training. Above 0.2 = significant drift.
- **"Why a Flask app instead of `mlflow models serve`?"** — MLflow's built-in server was
  incompatible with the newer library versions on Python 3.13, so I wrapped the *same* registered
  model in a small Flask service exposing the same API. Pragmatic, and it loads the identical
  Production model.
- **"What does the Airflow DAG do?"** — Orchestrates ingest → train → tune → register on a
  schedule, calling the same code, so the pipeline can re-run automatically.
- **"Where's the backend database?"** — MLflow stores run metadata in a SQLite database
  (`mlflow.db`) and artifacts (models, plots) in a local folder — both configured on the tracking
  server.

## If a demo fails (backup plan)
Have the **report** (`reports/REPORT.md`) and its **figures** open in another tab. If the UI or a
command misbehaves, narrate from the results table and the ROC/confusion-matrix images instead —
the story is identical.
