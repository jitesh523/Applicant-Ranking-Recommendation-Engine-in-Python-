import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb

DATA_PATH = Path(__file__).parent / "AI_Resume_Screening.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    # Normalize skills: lowercase and strip spaces
    df["Skills"] = (
        df["Skills"].astype(str)
        .str.lower()
        .str.replace(" ", "", regex=False)
    )
    return df


def build_role_skill_profiles(df: pd.DataFrame) -> pd.Series:
    """Build a simple text profile per Job Role by concatenating all skills of that role."""
    role_profiles = (
        df.groupby("Job Role")["Skills"]
        .apply(lambda s: " ".join(s.tolist()))
    )
    return role_profiles


def compute_skill_features(df: pd.DataFrame, role_profiles: pd.Series):
    # TF-IDF on skills (candidate skills + role skill profiles)
    all_docs = df["Skills"].tolist() + role_profiles.tolist()

    tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(","), lowercase=False)
    tfidf_matrix = tfidf.fit_transform(all_docs)

    n_candidates = len(df)
    candidate_tfidf = tfidf_matrix[:n_candidates]
    role_tfidf = tfidf_matrix[n_candidates:]

    # Map Job Role to index in role_profiles
    role_to_idx = {role: i for i, role in enumerate(role_profiles.index)}

    # For each candidate, compute cosine similarity between its skills and its role profile
    sims = []
    for i, role in enumerate(df["Job Role"].values):
        j = role_to_idx[role]
        sim = cosine_similarity(candidate_tfidf[i], role_tfidf[j])[0, 0]
        sims.append(sim)

    df["skill_profile_cosine"] = sims

    # Simple overlap counts between candidate skills and role-level required skills (approximated)
    # Build set of skills per role
    role_skill_sets = (
        df.groupby("Job Role")["Skills"]
        .apply(lambda s: set(",".join(s).split(",")))
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
    # Numeric features
    num_cols = [
        "Experience (Years)",
        "Salary Expectation ($)",
        "Projects Count",
        "skill_profile_cosine",
        "num_matching_skills",
        "skill_match_ratio",
    ]
    X_num = df[num_cols].astype(float).to_numpy()

    # Categorical features
    cat_cols = ["Education", "Certifications"]
    X_cat_input = df[cat_cols].fillna("Unknown").astype(str)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_cat = ohe.fit_transform(X_cat_input)

    # Label
    y = df["AI Score (0-100)"].astype(float).to_numpy()

    # Group by Job Role
    job_roles = df["Job Role"].astype(str)

    X = np.hstack([X_num, X_cat])

    return X, y, job_roles.to_numpy(), ohe, num_cols, cat_cols


def build_groups(indices, job_roles):
    role_counts = {}
    for i in indices:
        role = job_roles[i]
        role_counts[role] = role_counts.get(role, 0) + 1
    return [cnt for _, cnt in role_counts.items()]


def train_ranker_with_skills():
    df = load_data()
    role_profiles = build_role_skill_profiles(df)
    df = compute_skill_features(df, role_profiles)

    X, y, job_roles, ohe, num_cols, cat_cols = build_features(df)

    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    train_groups = build_groups(train_idx, job_roles)
    val_groups = build_groups(val_idx, job_roles)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )

    ranker.fit(
        X_train,
        y_train,
        group=train_groups,
        eval_set=[(X_val, y_val)],
        eval_group=[val_groups],
        eval_at=[5, 10],
        eval_metric="ndcg",
        early_stopping_rounds=30,
        verbose=True,
    )

    # Sample ranking for one role
    sample_role = df["Job Role"].value_counts().idxmax()
    mask = df["Job Role"] == sample_role
    X_role = X[mask]
    df_role = df[mask].copy()
    scores = ranker.predict(X_role)
    df_role["model_score"] = scores
    df_role_sorted = df_role.sort_values("model_score", ascending=False)

    print(f"\n=== Sample ranking for Job Role (with skills features): {sample_role} ===")
    print(
        df_role_sorted[
            [
                "Resume_ID",
                "Name",
                "Job Role",
                "AI Score (0-100)",
                "skill_profile_cosine",
                "num_matching_skills",
                "skill_match_ratio",
                "model_score",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    train_ranker_with_skills()
