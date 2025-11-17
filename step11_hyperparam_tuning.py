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


def add_skill_features(df: pd.DataFrame) -> pd.DataFrame:
    role_profiles = df.groupby("Job Role")["Skills"].apply(lambda s: " ".join(s.tolist()))

    all_docs = df["Skills"].tolist() + role_profiles.tolist()

    tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(","), lowercase=False)
    tfidf_matrix = tfidf.fit_transform(all_docs)

    n_candidates = len(df)
    cand_tfidf = tfidf_matrix[:n_candidates]
    role_tfidf = tfidf_matrix[n_candidates:]

    role_to_idx = {role: i for i, role in enumerate(role_profiles.index)}

    sims = []
    for i, role in enumerate(df["Job Role"].values):
        j = role_to_idx[role]
        sim = cosine_similarity(cand_tfidf[i], role_tfidf[j])[0, 0]
        sims.append(sim)
    df["skill_profile_cosine"] = sims

    role_skill_sets = df.groupby("Job Role")["Skills"].apply(lambda s: set(",".join(s).split(",")))

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
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = ohe.fit_transform(X_cat_input)

    raw_scores = df["AI Score (0-100)"].astype(float).to_numpy()
    bins = [0, 20, 40, 60, 80, 100]
    y = np.digitize(raw_scores, bins, right=True) - 1

    roles = df["Job Role"].astype(str).to_numpy()

    X = np.hstack([X_num, X_cat])
    return X, y, roles


def build_groups(indices, roles):
    counts = {}
    for i in indices:
        r = roles[i]
        counts[r] = counts.get(r, 0) + 1
    return [cnt for _, cnt in counts.items()]


def ndcg_at_k(relevances, k):
    rel = np.asarray(relevances)[:k]
    if rel.size == 0:
        return 0.0
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    ideal = np.sort(rel)[::-1]
    idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(2, ideal.size + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_config(df_val, roles_val, scores, y_val, k=5):
    ndcgs = []
    for role in np.unique(roles_val):
        mask = roles_val == role
        rel = y_val[mask]
        sc = scores[mask]
        order = np.argsort(-sc)
        rel_sorted = rel[order]
        ndcgs.append(ndcg_at_k(rel_sorted, k))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def random_search_hyperparams(X_train, y_train, roles_train, X_val, y_val, roles_val, n_trials=10):
    best_score = -1.0
    best_params = None

    train_groups = build_groups(np.arange(len(X_train)), roles_train)
    val_groups = build_groups(np.arange(len(X_val)), roles_val)

    for i in range(n_trials):
        params = {
            "num_leaves": int(np.random.choice([15, 31, 63, 127])),
            "learning_rate": float(np.random.choice([0.01, 0.03, 0.05, 0.1])),
            "n_estimators": int(np.random.choice([100, 200, 400])),
        }
        model = lgb.LGBMRanker(
            objective="lambdarank",
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            random_state=42,
        )
        model.fit(
            X_train,
            y_train,
            group=train_groups,
            eval_set=[(X_val, y_val)],
            eval_group=[val_groups],
            eval_at=[5],
            eval_metric="ndcg",
            verbose=False,
        )
        scores = model.predict(X_val)
        ndcg5 = evaluate_config(None, roles_val, scores, y_val, k=5)
        print(f"Trial {i+1}/{n_trials} params={params} NDCG@5={ndcg5:.4f}")
        if ndcg5 > best_score:
            best_score = ndcg5
            best_params = params

    return best_params, best_score


def main():
    df = load_data()
    df = add_skill_features(df)

    X, y, roles = build_features(df)

    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    X_train, y_train, roles_train = X[train_idx], y[train_idx], roles[train_idx]
    X_val, y_val, roles_val = X[val_idx], y[val_idx], roles[val_idx]

    best_params, best_score = random_search_hyperparams(X_train, y_train, roles_train, X_val, y_val, roles_val, n_trials=10)

    print("\n=== Best hyperparameters found ===")
    print(best_params)
    print(f"Best NDCG@5: {best_score:.4f}")


if __name__ == "__main__":
    main()
