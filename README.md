<div align="center">

# Applicant Ranking & Recommendation Engine

Learning‑to‑rank pipeline in Python for scoring and ranking job applicants, inspired by HR‑tech platforms like **Phenom**.

</div>

---

## 1. Overview

This repository contains an end‑to‑end **applicant ranking and recommendation engine** built around a synthetic resume dataset.

Each row in `AI_Resume_Screening.csv` represents one candidate with:

- **Skills** (comma‑separated)
- **Experience (Years)**
- **Education** (B.Sc, B.Tech, M.Tech, MBA, PhD, …)
- **Certifications**
- **Job Role** (AI Researcher, Data Scientist, Software Engineer, Cybersecurity Analyst)
- **Recruiter Decision** (`Hire` / `Reject`)
- **AI Score (0–100)** – a numeric relevance/quality score
- Salary expectation & project count

We treat each **Job Role** as a query and learn to **rank candidates within each role** using feature engineering + **LightGBM’s LGBMRanker** (LambdaRank).

The project is organized as small, incremental scripts so you can walk through the system step‑by‑step.

---

## 2. Features at a Glance

- **Dataset exploration**: understand columns, distributions, and label quality.
- **Baseline ranker**: numeric + categorical features only.
- **Skill‑aware ranker**:
  - Role‑level skill profiles.
  - Skill overlap counts and ratios.
  - TF‑IDF cosine similarity between candidate skills and role profile.
- **Job description similarity**:
  - Synthetic `job_descriptions.csv`.
  - TF‑IDF similarity between job description text and candidate skills.
- **Labels**:
  - Raw `AI Score (0–100)`.
  - Discretized relevance levels (0–4).
  - Binary relevance from `Recruiter Decision` (Hire/Reject).
- **Learning‑to‑rank** with **LightGBM** (LambdaRank objective).
- **Evaluation**: NDCG@K, Precision@K, MRR.
- **Explainability**: global feature importance using **SHAP**.
- **Fairness proxy**: compares education groups in model outputs vs label quality.
- **Hyperparameter tuning**: simple random search over LightGBM ranker params.
- **Streamlit UI**: interactive web app to select a role and see ranked candidates.

---

## 3. Project Structure

Key files:

- `AI_Resume_Screening.csv` – main resume dataset.
- `job_descriptions.csv` – synthetic job description text per `Job Role`.
- `requirements.txt` – Python dependencies.
- `README.md` – this documentation.

Step‑by‑step scripts:

- **Exploration & baselines**
  - `step1_load_and_explore.py` – basic EDA.
  - `step2_basic_ranker.py` – baseline ranker (no skill features).

- **Skill‑aware learning‑to‑rank**
  - `step3_ranker_with_skills.py` – ranker with skill overlap + cosine similarity.
  - `step4_evaluate_ranking.py` – NDCG/Precision/MRR evaluation.
  - `step7_ranker_with_hire_label.py` – ranker using `Recruiter Decision` as binary relevance.

- **Explainability**
  - `step6_explain_with_shap.py` – SHAP explanations for feature importance.

- **Job‑description features**
  - `step9_ranker_with_job_desc.py` – adds JD–skills cosine similarity and evaluates.

- **Fairness & tuning**
  - `step10_fairness_checks.py` – fairness proxy across education groups.
  - `step11_hyperparam_tuning.py` – random search over ranker hyperparameters.

- **Pipeline & UI**
  - `step5_ranking_pipeline.py` – reusable `RankingPipeline` class for training + inference.
  - `step8_streamlit_app.py` – Streamlit app on top of `RankingPipeline`.

---

## 4. Setup

### 4.1. Python environment

Use Python 3.9+ if possible.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

This installs (among others):

- `pandas`, `numpy`
- `scikit-learn`
- `lightgbm`
- `streamlit`
- `shap`
- `jupyter`

> If you get PATH warnings for scripts, you can ignore them or add the suggested directory to your PATH.

---

## 5. Step‑by‑Step Scripts

You can run each step independently to understand the pipeline.

### 5.1. Step 1 – Explore the dataset

```bash
python3 step1_load_and_explore.py
```

Prints:

- Schema (`df.info()`)
- Head of the data
- `Job Role` distribution
- `Recruiter Decision` distribution
- Summary stats for `AI Score (0–100)`

---

### 5.2. Step 2 – Baseline ranker

```bash
python3 step2_basic_ranker.py
```

Features:

- Numeric: `Experience (Years)`, `Salary Expectation ($)`, `Projects Count`
- Categorical: `Education`, `Certifications` (OneHotEncoder)

Label: `AI Score (0–100)` (continuous).

Model: `LGBMRanker(objective="lambdarank")` grouped by `Job Role`.

Outputs a sample top‑10 ranking for the most common role.

---

### 5.3. Step 3 – Skill‑aware ranker

```bash
python3 step3_ranker_with_skills.py
```

Adds skill features:

- Normalized `Skills` (lowercase, no spaces).
- Role‑level skill profiles (concatenated skills per `Job Role`).
- `skill_profile_cosine` (TF‑IDF cosine similarity between candidate skills and role profile).
- `num_matching_skills`, `skill_match_ratio`.

Trains `LGBMRanker` and prints a ranked list for a sample role.

---

### 5.4. Step 4 – Ranking evaluation

```bash
python3 step4_evaluate_ranking.py
```

Reuses skill features and evaluates on a validation split:

- `NDCG@3`, `NDCG@5`, `NDCG@10`
- `Precision@3`, `Precision@5`, `Precision@10`
- `MRR` (Mean Reciprocal Rank)

High `NDCG` means relevant candidates are near the top; high `MRR` means the first good candidate is ranked high.

---

### 5.5. Step 6 – SHAP explainability

```bash
python3 step6_explain_with_shap.py
```

Trains a `LGBMRegressor` on the same feature set and uses **SHAP** to:

- Compute SHAP values for a sample of candidates.
- Print the top 10 most important features by mean absolute SHAP value.
- Optionally generate a SHAP summary plot (works best in a Jupyter notebook or GUI environment).

---

### 5.6. Step 7 – Ranker with Hire/Reject label

```bash
python3 step7_ranker_with_hire_label.py
```

Uses the same skill features but defines relevance as:

- `Recruiter Decision == "Hire"` → 1
- `Recruiter Decision == "Reject"` → 0

Trains `LGBMRanker` and evaluates NDCG/Precision/MRR using this binary label.

---

### 5.7. Step 9 – Job description similarity

```bash
python3 step9_ranker_with_job_desc.py
```

Adds job description similarity features using `job_descriptions.csv`:

- Merges JD text per `Job Role`.
- Computes `jd_skills_cosine` via TF‑IDF between candidate `Skills` and `JobDescription`.

Label: discretized `AI Score (0–100)` into 5 relevance levels (0–4).

Evaluates ranking with the additional JD feature.

---

### 5.8. Step 10 – Fairness proxy across education

```bash
python3 step10_fairness_checks.py
```

Trains a skill‑aware ranker and prints, per `Education` level:

- `frac_high_label` – fraction of candidates in that group whose relevance label is high (e.g., ≥3).
- `frac_in_top20` – fraction of the model’s top‑20 ranked candidates that come from that group.

This gives a rough view of whether certain education groups are over‑ or under‑represented in top recommendations relative to their labels.

---

### 5.9. Step 11 – Hyperparameter tuning

```bash
python3 step11_hyperparam_tuning.py
```

Performs a simple **random search** over LightGBM ranker hyperparameters:

- `num_leaves` ∈ {15, 31, 63, 127}
- `learning_rate` ∈ {0.01, 0.03, 0.05, 0.1}
- `n_estimators` ∈ {100, 200, 400}

For each trial it prints:

```text
Trial i/n params={...} NDCG@5=...
```

and finally prints the best configuration and best NDCG@5.

You can plug the best params into other scripts or into `RankingPipeline`.

---

## 6. Reusable Pipeline & Streamlit App

### 6.1. RankingPipeline class

File: `step5_ranking_pipeline.py`

Defines `RankingPipeline` for training and inference:

- `fit(df)`:
  - Normalizes `Skills`.
  - Builds role skill profiles and TF‑IDF model.
  - Computes `skill_profile_cosine`, `num_matching_skills`, `skill_match_ratio`.
  - One‑hot encodes `Education`, `Certifications`.
  - Converts `AI Score (0–100)` into 5 relevance levels (0–4).
  - Trains a single `LGBMRanker` grouped by `Job Role`.

- `rank_candidates_for_role(job_role, candidates_df)`:
  - Computes the same features for a given set of candidates and role.
  - Returns a DataFrame sorted by `model_score` (descending).

You can run a simple demo:

```bash
python3 step5_ranking_pipeline.py
```

### 6.2. Streamlit app

File: `step8_streamlit_app.py`

Launch with:

```bash
python3 -m streamlit run step8_streamlit_app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

The app lets you:

- Select a **Job Role** from a dropdown.
- Click **Rank candidates** to run `RankingPipeline` and see top candidates.
- Inspect key fields per candidate:
  - `Name`, `Resume_ID`, `Job Role`, `Experience (Years)`
  - `Skills`, `Education`, `Certifications`
  - `AI Score (0–100)`
  - `skill_profile_cosine`, `num_matching_skills`, `skill_match_ratio`
  - `model_score`
- Drill into a single candidate via their `Resume_ID`.

---

## 7. Possible Extensions

Ideas to extend this project:

- Incorporate **real job descriptions** and richer text features (e.g., embeddings).
- Add a proper **two‑stage retrieval + ranking** architecture.
- Integrate **LLM‑based skill extraction** or semantic matching.
- Build more sophisticated **fairness metrics** (e.g., using `fairlearn`).
- Turn this into a small **web service** (FastAPI/Fastify) behind the Streamlit UI.

---

## 8. License

Add your preferred license here (e.g., MIT) if you intend to make the project public.

