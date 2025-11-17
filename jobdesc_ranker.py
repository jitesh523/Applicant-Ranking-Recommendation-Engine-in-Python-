import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb

DATA_PATH = Path(__file__).parent / "AI_Resume_Screening.csv"
JD_PATH = Path(__file__).parent / "job_descriptions.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df["Skills"] = (
        df["Skills"].astype(str)
        .str.lower()
        .str.replace(" ", "", regex=False)
    )
    jd = pd.read_csv(JD_PATH)
    jd.columns = [c.strip() for c in jd.columns]
    return df, jd


def add_job_desc_similarity(df: pd.DataFrame, jd: pd.DataFrame) -> pd.DataFrame:
    # Merge job descriptions onto candidates by Job Role
    df = df.merge(jd, on="Job Role", how="left")

    # Build TF-IDF over combined documents: candidate skills + job description text
    # Represent skills and JDs as simple text
    cand_docs = df["Skills"].fillna("")
    jd_docs = df["JobDescription"].fillna("").str.lower()

    all_docs = pd.concat([cand_docs, jd_docs], axis=0).tolist()

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(all_docs)

    n = len(df)
    cand_tfidf = tfidf_matrix[:n]
    jd_tfidf = tfidf_matrix[n:]

    sims = []
    for i in range(n):
        sim = cosine_similarity(cand_tfidf[i], jd_tfidf[i])[0, 0]
        sims.append(sim)

    df["jd_skills_cosine"] = sims
    return df


def build_features(df: pd.DataFrame):
    num_cols = [
        "Experience (Years)",
        "Salary Expectation ($)",
        "Projects Count",
        "jd_skills_cosine",
    ]
    X_num = df[num_cols].astype(float).to_numpy()

    cat_cols = ["Education", "Certifications", "Job Role"]
    X_cat_input = df[cat_cols].fillna("Unknown").astype(str)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = ohe.fit_transform(X_cat_input)

    # Relevance: discretized AI Score (0-100) into 0-4
    raw_scores = df["AI Score (0-100)"].astype(float).to_numpy()
    bins = [0, 20, 40, 60, 80, 100]
    y = np.digitize(raw_scores, bins, right=True) - 1

    job_roles = df["Job Role"].astype(str).to_numpy()

    X = np.hstack([X_num, X_cat])
    return X, y, job_roles


def build_groups(indices, roles):
    role_counts = {}
    for i in indices:
        r = roles[i]
        role_counts[r] = role_counts.get(r, 0) + 1
    return [cnt for _, cnt in role_counts.items()]


def ndcg_at_k(relevances, k):
    rel = np.asarray(relevances)[:k]
    if rel.size == 0:
        return 0.0
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    ideal = np.sort(rel)[::-1]
    idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(2, ideal.size + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def precision_at_k(relevances, k, threshold=3):
    rel = np.asarray(relevances)[:k]
    hits = np.sum(rel >= threshold)
    return float(hits) / max(1, len(rel))


def mrr(ranks):
    if not ranks:
        return 0.0
    return float(np.mean([1.0 / r for r in ranks]))


def evaluate(df, roles, scores, y, k_list=(3, 5, 10), high_relevance_level=3):
    metrics = {f"ndcg@{k}": [] for k in k_list}
    metrics.update({f"precision@{k}": [] for k in k_list})
    mrr_ranks = []

    for role in np.unique(roles):
        mask = roles == role
        df_role = df[mask].copy()
        role_scores = scores[mask]
        role_y = y[mask]

        order = np.argsort(-role_scores)
        role_y_sorted = role_y[order]

        for k in k_list:
            metrics[f"ndcg@{k}"].append(ndcg_at_k(role_y_sorted, k))
            metrics[f"precision@{k}"].append(precision_at_k(role_y_sorted, k, threshold=high_relevance_level))

        hits = np.where(role_y_sorted >= high_relevance_level)[0]
        if hits.size > 0:
            mrr_ranks.append(hits[0] + 1)

    summary = {}
    for k in k_list:
        summary[f"ndcg@{k}"] = float(np.mean(metrics[f"ndcg@{k}"])) if metrics[f"ndcg@{k}"] else 0.0
        summary[f"precision@{k}"] = float(np.mean(metrics[f"precision@{k}"])) if metrics[f"precision@{k}"] else 0.0
    summary["mrr"] = mrr(mrr_ranks)
    return summary


def main():
    df, jd = load_data()
    df = add_job_desc_similarity(df, jd)

    X, y, roles = build_features(df)

    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    X_train, y_train, roles_train = X[train_idx], y[train_idx], roles[train_idx]
    X_val, y_val, roles_val = X[val_idx], y[val_idx], roles[val_idx]

    train_groups = build_groups(train_idx, roles)
    val_groups = build_groups(val_idx, roles)

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

    val_scores = ranker.predict(X_val)
    df_val = df.iloc[val_idx].copy()

    summary = evaluate(df_val, roles_val, val_scores, y_val, k_list=(3, 5, 10), high_relevance_level=3)

    print("\n=== Ranking evaluation with Job Description similarity feature ===")
    for k in (3, 5, 10):
        print(f"NDCG@{k}: {summary[f'ndcg@{k}']:.4f}")
        print(f"Precision@{k}: {summary[f'precision@{k}']:.4f}")
    print(f"MRR: {summary['mrr']:.4f}")


if __name__ == "__main__":
    main()
