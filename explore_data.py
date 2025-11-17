import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent / "AI_Resume_Screening.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Basic normalization: strip spaces from column names
    df.columns = [c.strip() for c in df.columns]
    return df


def main() -> None:
    df = load_data()

    print("=== Basic Info ===")
    print(df.info())
    print()

    print("=== Head ===")
    print(df.head())
    print()

    print("=== Job Role distribution ===")
    print(df["Job Role"].value_counts())
    print()

    print("=== Recruiter Decision distribution ===")
    print(df["Recruiter Decision"].value_counts())
    print()

    print("=== AI Score stats ===")
    print(df["AI Score (0-100)"].describe())


if __name__ == "__main__":
    main()
