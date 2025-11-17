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
    df["Skills"] = (
        df["Skills"].astype(str)
        .str.lower()
        .str.replace(" ", "", regex=False)
    )
    return df


def build_role_skill_profiles(df: pd.DataFrame) -> pd.Series:
    role_profiles = (
        df.groupby("Job Role")["Skills"]
        .apply(lambda s: " ".join(s.tolist()))
    )
    return role_profiles


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

    y = df["AI Score (0-100)"].astype(float).to_numpy()
    job_roles = df["Job Role"].astype(str).to_numpy()

    X = np.hstack([X_num, X_cat])
    return X, y, job_roles


def split_train_val(X, y, job_roles, test_size=0.2, random_state=42):
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
    return train_idx, val_idx


def train_ranker(X_train, y_train, train_roles, X_val, y_val, val_roles):
    def build_groups(indices, roles):
        role_counts = {}
        for i in indices:
            role = roles[i]
            role_counts[role] = role_counts.get(role, 0) + 1
        return [cnt for _, cnt in role_counts.items()]

    train_groups = build_groups(np.arange(len(X_train)), train_roles)
    val_groups = build_groups(np.arange(len(X_val)), val_roles)

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

    return ranker


def ndcg_at_k(relevances, k):
    rel = np.asarray(relevances)[:k]
    if rel.size == 0:
        return 0.0
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    ideal = np.sort(rel)[::-1]
    idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(2, ideal.size + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def mrr(ranks):
    # ranks: list of ranks (1-based) of the first relevant item per query
    if not ranks:
        return 0.0
    return float(np.mean([1.0 / r for r in ranks]))


def precision_at_k(relevances, k, threshold=80.0):
    rel = np.asarray(relevances)[:k]
    hits = np.sum(rel >= threshold)
    return float(hits) / max(1, len(rel))


def evaluate_model(df, job_roles, scores, k_list=(3, 5, 10), relevance_threshold=80.0):
    metrics = {f"ndcg@{k}": [] for k in k_list}
    metrics.update({f"precision@{k}": [] for k in k_list})
    mrr_ranks = []

    for role in np.unique(job_roles):
        mask = job_roles == role
        df_role = df[mask].copy()
        role_scores = scores[mask]

        # Sort by predicted score
        order = np.argsort(-role_scores)
        df_role = df_role.iloc[order]

        relevances = df_role["AI Score (0-100)"].to_numpy()

        # NDCG and Precision@K
        for k in k_list:
            metrics[f"ndcg@{k}"].append(ndcg_at_k(relevances, k))
            metrics[f"precision@{k}"].append(precision_at_k(relevances, k, threshold=relevance_threshold))

        # MRR: rank of first highly relevant (>= threshold)
        hits = np.where(relevances >= relevance_threshold)[0]
        if hits.size > 0:
            mrr_ranks.append(hits[0] + 1)  # 1-based

    # Aggregate
    summary = {}
    for k in k_list:
        summary[f"ndcg@{k}"] = float(np.mean(metrics[f"ndcg@{k}"])) if metrics[f"ndcg@{k}"] else 0.0
        summary[f"precision@{k}"] = float(np.mean(metrics[f"precision@{k}"])) if metrics[f"precision@{k}"] else 0.0
    summary["mrr"] = mrr(mrr_ranks)

    return summary


def main():
    df = load_data()
    role_profiles = build_role_skill_profiles(df)
    df = compute_skill_features(df, role_profiles)

    X, y, job_roles = build_features(df)

    train_idx, val_idx = split_train_val(X, y, job_roles)

    X_train, y_train, roles_train = X[train_idx], y[train_idx], job_roles[train_idx]
    X_val, y_val, roles_val = X[val_idx], y[val_idx], job_roles[val_idx]

    ranker = train_ranker(X_train, y_train, roles_train, X_val, y_val, roles_val)

    # Evaluate on validation set
    val_scores = ranker.predict(X_val)
    df_val = df.iloc[val_idx].copy()

    summary = evaluate_model(df_val, roles_val, val_scores, k_list=(3, 5, 10), relevance_threshold=80.0)

    print("\n=== Ranking evaluation on validation set ===")
    for k in (3, 5, 10):
        print(f"NDCG@{k}: {summary[f'ndcg@{k}']:.4f}")
        print(f"Precision@{k}: {summary[f'precision@{k}']:.4f}")
    print(f"MRR: {summary['mrr']:.4f}")


if __name__ == "__main__":
    main()
