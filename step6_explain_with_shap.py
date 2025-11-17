import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb
import shap

DATA_PATH = Path(__file__).parent / "AI_Resume_Screening.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df["Skills"] = (
        df["Skills"].astype(str)
        .str.lower()
        .str.replace(" ", "", regex=False)
    )
    return df


def build_role_skill_profiles(df: pd.DataFrame) -> pd.Series:
    return df.groupby("Job Role")["Skills"].apply(lambda s: " ".join(s.tolist()))


def compute_skill_features(df: pd.DataFrame, role_profiles: pd.Series):
    all_docs = df["Skills"].tolist() + role_profiles.tolist()

    tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(","), lowercase=False)
    tfidf_matrix = tfidf.fit_transform(all_docs)

    n_candidates = len(df)
    candidate_tfidf = tfidf_matrix[:n_candidates]
    role_tfidf = tfidf_matrix[n_candidates:]

    role_to_idx = {role: i for i, role in enumerate(role_profiles.index)}

    sims = []
    for i, role in enumerate(df["Job Role"].values):
        j = role_to_idx[role]
        sim = cosine_similarity(candidate_tfidf[i], role_tfidf[j])[0, 0]
        sims.append(sim)
    df["skill_profile_cosine"] = sims

    role_skill_sets = (
        df.groupby("Job Role")["Skills"].apply(lambda s: set(",".join(s).split(",")))
    )

    num_match = []
    ratio_match = []
    for skills_str, role in zip(df["Skills"].values, df["Job Role"].values):
        cand_set = set(skills_str.split(",")) if skills_str else set()
        role_set = role_skill_sets[role]
        inter = cand_set & role_set
        num_match.append(len(inter))
        ratio_match.append(len(inter) / max(1, len(role_set)))

    df["num_matching_skills"] = num_match
    df["skill_match_ratio"] = ratio_match

    return df


def build_features(df: pd.DataFrame):
    num_cols = [
        "Experience (Years)",
        "Salary Expectation ($)",
        "Projects Count",
        "skill_profile_cosine",
        "num_matching_skills",
        "skill_match_ratio",
    ]
    X_num = df[num_cols].astype(float).to_numpy()

    cat_cols = ["Education", "Certifications"]
    X_cat_input = df[cat_cols].fillna("Unknown").astype(str)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_cat = ohe.fit_transform(X_cat_input)

    feature_names = num_cols + [
        f"ohe_{col}_{val}" for col, vals in zip(cat_cols, ohe.categories_) for val in vals
    ]

    X = np.hstack([X_num, X_cat])
    y = df["AI Score (0-100)"].astype(float).to_numpy()
    return X, y, feature_names


def train_ranker(X, y):
    # Here we ignore grouping for simplicity since we just want a model for SHAP explanation.
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    model.fit(X, y)
    return model


def explain_with_shap():
    df = load_data()
    role_profiles = build_role_skill_profiles(df)
    df = compute_skill_features(df, role_profiles)

    X, y, feature_names = build_features(df)

    model = train_ranker(X, y)

    explainer = shap.TreeExplainer(model)

    # Sample a subset of candidates for speed
    sample_idx = np.random.choice(len(X), size=min(50, len(X)), replace=False)
    X_sample = X[sample_idx]

    shap_values = explainer.shap_values(X_sample)

    # Summary plot (will open a window in some environments or can be saved)
    print("Computing SHAP summary plot... (this may open a window or require a notebook)")
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)

    # Print top features (global importance) numerically as well
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(-mean_abs)
    print("\n=== Top 10 most important features by SHAP ===")
    for i in order[:10]:
        print(f"{feature_names[i]}: {mean_abs[i]:.4f}")


if __name__ == "__main__":
    explain_with_shap()
