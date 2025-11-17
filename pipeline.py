import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb

DATA_PATH = Path(__file__).parent / "AI_Resume_Screening.csv"


class RankingPipeline:
    def __init__(self):
        self.ohe = None
        self.tfidf = None
        self.role_profiles = None
        self.role_skill_sets = None
        self.model = None
        self.num_cols = [
            "Experience (Years)",
            "Salary Expectation ($)",
            "Projects Count",
            "skill_profile_cosine",
            "num_matching_skills",
            "skill_match_ratio",
        ]
        self.cat_cols = ["Education", "Certifications"]

    def _normalize_skills(self, s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
            .str.lower()
            .str.replace(" ", "", regex=False)
        )

    def _build_role_profiles_and_tfidf(self, df: pd.DataFrame):
        df["Skills"] = self._normalize_skills(df["Skills"])

        self.role_profiles = (
            df.groupby("Job Role")["Skills"]
            .apply(lambda s: " ".join(s.tolist()))
        )

        # Build TF-IDF over candidate skills + role profiles
        all_docs = df["Skills"].tolist() + self.role_profiles.tolist()
        self.tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(","), lowercase=False)
        tfidf_matrix = self.tfidf.fit_transform(all_docs)

        n_candidates = len(df)
        candidate_tfidf = tfidf_matrix[:n_candidates]
        role_tfidf = tfidf_matrix[n_candidates:]

        role_to_idx = {role: i for i, role in enumerate(self.role_profiles.index)}

        sims = []
        for i, role in enumerate(df["Job Role"].values):
            j = role_to_idx[role]
            sim = cosine_similarity(candidate_tfidf[i], role_tfidf[j])[0, 0]
            sims.append(sim)
        df["skill_profile_cosine"] = sims

        self.role_skill_sets = (
            df.groupby("Job Role")["Skills"]
            .apply(lambda s: set(",".join(s).split(",")))
        )

        return df

    def _add_skill_overlap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        num_match = []
        ratio_match = []
        for skills_str, role in zip(df["Skills"].values, df["Job Role"].values):
            cand_set = set(skills_str.split(",")) if skills_str else set()
            role_set = self.role_skill_sets.get(role, set())
            inter = cand_set & role_set
            num_match.append(len(inter))
            ratio_match.append(len(inter) / max(1, len(role_set)))

        df["num_matching_skills"] = num_match
        df["skill_match_ratio"] = ratio_match
        return df

    def _build_ohe(self, df: pd.DataFrame):
        X_cat_input = df[self.cat_cols].fillna("Unknown").astype(str)
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohe.fit(X_cat_input)

    def _transform_features(self, df: pd.DataFrame) -> np.ndarray:
        X_num = df[self.num_cols].astype(float).to_numpy()
        X_cat_input = df[self.cat_cols].fillna("Unknown").astype(str)
        X_cat = self.ohe.transform(X_cat_input)
        return np.hstack([X_num, X_cat])

    def fit(self, df: pd.DataFrame):
        # Assumes df is the full training dataset
        df = df.copy()
        df["Skills"] = self._normalize_skills(df["Skills"])

        # Build profiles, tfidf, and skill_profile_cosine
        df = self._build_role_profiles_and_tfidf(df)
        # Add skill overlap features (depends on role_skill_sets)
        df = self._add_skill_overlap_features(df)

        # Build encoder
        self._build_ohe(df)

        # Features and labels
        X = self._transform_features(df)

        # Convert continuous AI Score (0-100) into discrete relevance levels 0-4
        raw_scores = df["AI Score (0-100)"].astype(float).to_numpy()
        # Bins: [0,20] -> 0, (20,40] -> 1, (40,60] -> 2, (60,80] -> 3, (80,100] -> 4
        bins = [0, 20, 40, 60, 80, 100]
        y = np.digitize(raw_scores, bins, right=True) - 1

        # Group by Job Role
        roles = df["Job Role"].astype(str).to_numpy()
        role_counts = {}
        for r in roles:
            role_counts[r] = role_counts.get(r, 0) + 1
        groups = [cnt for _, cnt in role_counts.items()]

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
        )

        self.model.fit(
            X,
            y,
            group=groups,
        )

    def rank_candidates_for_role(self, job_role: str, candidates_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Pipeline model is not fitted. Call fit() first.")

        df = candidates_df.copy()
        df["Job Role"] = job_role
        df["Skills"] = self._normalize_skills(df["Skills"])

        # Compute skill_profile_cosine using existing TF-IDF and role profiles
        all_docs = df["Skills"].tolist() + self.role_profiles.tolist()
        tfidf_matrix = self.tfidf.transform(all_docs)
        n_candidates = len(df)
        candidate_tfidf = tfidf_matrix[:n_candidates]
        role_tfidf = tfidf_matrix[n_candidates:]

        role_to_idx = {role: i for i, role in enumerate(self.role_profiles.index)}
        sims = []
        for i, _ in enumerate(df["Job Role"].values):
            j = role_to_idx[job_role]
            sim = cosine_similarity(candidate_tfidf[i], role_tfidf[j])[0, 0]
            sims.append(sim)
        df["skill_profile_cosine"] = sims

        # Add overlap features using stored role_skill_sets
        df = self._add_skill_overlap_features(df)

        # Transform to feature matrix and predict
        X = self._transform_features(df)
        scores = self.model.predict(X)

        df["model_score"] = scores
        df_sorted = df.sort_values("model_score", ascending=False)
        return df_sorted


def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def demo():
    df = load_training_data()
    pipeline = RankingPipeline()
    pipeline.fit(df)

    # Demo: rank all Data Scientist candidates for the Data Scientist role
    role = "Data Scientist"
    candidates = df[df["Job Role"] == role].copy()
    ranked = pipeline.rank_candidates_for_role(role, candidates)

    print(f"\n=== Demo ranking for role: {role} ===")
    print(
        ranked[[
            "Resume_ID",
            "Name",
            "Job Role",
            "AI Score (0-100)",
            "skill_profile_cosine",
            "num_matching_skills",
            "skill_match_ratio",
            "model_score",
        ]].head(10)
    )


if __name__ == "__main__":
    demo()
