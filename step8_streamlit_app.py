import streamlit as st
import pandas as pd
from pathlib import Path

from step5_ranking_pipeline import RankingPipeline, load_training_data

DATA_PATH = Path(__file__).parent / "AI_Resume_Screening.csv"


@st.cache_resource
def load_pipeline() -> RankingPipeline:
    df = load_training_data()
    pipeline = RankingPipeline()
    pipeline.fit(df)
    return pipeline


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    st.title("Applicant Ranking & Recommendation Engine")
    st.write("Rank candidates by job role using a learning-to-rank model.")

    df = load_data()
    pipeline = load_pipeline()

    roles = sorted(df["Job Role"].unique().tolist())
    selected_role = st.selectbox("Select Job Role", roles)

    # Filter candidates for this role
    candidates = df[df["Job Role"] == selected_role].copy()

    st.subheader(f"Candidates for role: {selected_role}")
    st.write(f"Total candidates: {len(candidates)}")

    if candidates.empty:
        st.warning("No candidates for this role in the dataset.")
        return

    if st.button("Rank candidates"):
        ranked = pipeline.rank_candidates_for_role(selected_role, candidates)

        display_cols = [
            "Resume_ID",
            "Name",
            "Job Role",
            "Experience (Years)",
            "Skills",
            "Education",
            "Certifications",
            "AI Score (0-100)",
            "skill_profile_cosine",
            "num_matching_skills",
            "skill_match_ratio",
            "model_score",
        ]
        display_cols = [c for c in display_cols if c in ranked.columns]

        st.subheader("Ranked candidates (top 20)")
        st.dataframe(ranked[display_cols].head(20))

        # Optionally show details for a single candidate
        st.subheader("Inspect single candidate")
        resume_ids = ranked["Resume_ID"].tolist()
        selected_resume = st.selectbox("Select Resume ID", resume_ids)
        cand_row = ranked[ranked["Resume_ID"] == selected_resume]
        st.write(cand_row[display_cols].T)


if __name__ == "__main__":
    main()
