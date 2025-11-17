import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

DATA_PATH = Path(__file__).parent / "AI_Resume_Screening.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_features(df: pd.DataFrame):
    # Basic numeric features
    num_cols = [
        "Experience (Years)",
        "Salary Expectation ($)",
        "Projects Count",
    ]
    X_num = df[num_cols].astype(float).to_numpy()

    # Encode education and certifications and job role as categorical
    cat_cols = ["Education", "Certifications"]
    X_cat_input = df[cat_cols].fillna("Unknown").astype(str)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_cat = ohe.fit_transform(X_cat_input)

    # Simple label: AI Score (0-100) as relevance label
    y = df["AI Score (0-100)"].astype(float).to_numpy()

    # Groups: each Job Role is a query; group size is number of rows per role
    job_roles = df["Job Role"].astype(str)
    role_to_indices = {}
    for idx, role in enumerate(job_roles):
        role_to_indices.setdefault(role, []).append(idx)

    groups = [len(indices) for _, indices in role_to_indices.items()]

    X = np.hstack([X_num, X_cat])

    return X, y, np.array(groups), job_roles.to_numpy(), ohe, num_cols, cat_cols


def train_ranker():
    df = load_data()
    X, y, groups, job_roles, ohe, num_cols, cat_cols = build_features(df)

    # Train/validation split at candidate level, but keep group info recomputed for each set
    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    def build_group_for_indices(indices, job_roles_all):
        # recompute group sizes per job role using only the selected indices
        role_counts = {}
        for i in indices:
            role = job_roles_all[i]
            role_counts[role] = role_counts.get(role, 0) + 1
        return [cnt for _, cnt in role_counts.items()]

    train_groups = build_group_for_indices(train_idx, job_roles)
    val_groups = build_group_for_indices(val_idx, job_roles)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=200,
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
        eval_at=[5],
        eval_metric="ndcg",
        early_stopping_rounds=20,
        verbose=True,
    )

    # Quick sanity check: score a single job role and print top 5
    sample_role = df["Job Role"].value_counts().idxmax()
    mask = df["Job Role"] == sample_role
    X_role = X[mask]
    df_role = df[mask].copy()
    scores = ranker.predict(X_role)
    df_role["model_score"] = scores
    df_role_sorted = df_role.sort_values("model_score", ascending=False)

    print(f"\n=== Sample ranking for Job Role: {sample_role} ===")
    print(df_role_sorted[["Resume_ID", "Name", "Job Role", "AI Score (0-100)", "model_score"]].head(10))


if __name__ == "__main__":
    train_ranker()
