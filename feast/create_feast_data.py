import os
import pandas as pd


RAW_PATH = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT_PATH = "feast/data/employee_features.parquet"


def main():

    print("=" * 60)
    print("CREATING FEAST FEATURE DATA")
    print("=" * 60)

    # Load original dataset
    df = pd.read_csv(RAW_PATH)

    # Keep EmployeeNumber as Feast entity key
    feast_df = df[
        [
            "EmployeeNumber",
            "Age",
            "JobSatisfaction",
            "EnvironmentSatisfaction",
            "RelationshipSatisfaction",
            "YearsAtCompany",
            "YearsInCurrentRole",
            "YearsWithCurrManager",
            "YearsSinceLastPromotion"
        ]
    ].copy()

    # ---------------------------------------------------------
    # Engineered features
    # ---------------------------------------------------------

    feast_df["TotalExperience"] = (
        feast_df["YearsAtCompany"]
        + feast_df["YearsInCurrentRole"]
        + feast_df["YearsWithCurrManager"]
    )

    feast_df["PromotionRatio"] = (
        feast_df["YearsSinceLastPromotion"]
        / (feast_df["YearsAtCompany"] + 1)
    )

    feast_df["JobSatisfactionScore"] = (
        feast_df["JobSatisfaction"]
        + feast_df["EnvironmentSatisfaction"]
        + feast_df["RelationshipSatisfaction"]
    ) / 3

    feast_df["YearsPerCompanyAge"] = (
        feast_df["YearsAtCompany"]
        / (feast_df["Age"] + 1)
    )

    feast_df["CareerStagnationIndex"] = (
        feast_df["YearsAtCompany"]
        / (feast_df["YearsSinceLastPromotion"] + 1)
    )

    # Feast requires an event timestamp.
    # The original IBM dataset does not contain one, so we use
    # a fixed timestamp for this academic Feature Store example.
    feast_df["event_timestamp"] = pd.Timestamp(
        "2026-01-01",
        tz="UTC"
    )

    # Convert numeric columns to stable types
    numeric_columns = [
        "Age",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsWithCurrManager",
        "YearsSinceLastPromotion",
        "TotalExperience",
        "PromotionRatio",
        "JobSatisfactionScore",
        "YearsPerCompanyAge",
        "CareerStagnationIndex"
    ]

    for column in numeric_columns:
        feast_df[column] = pd.to_numeric(
            feast_df[column],
            errors="coerce"
        )

    os.makedirs("feast/data", exist_ok=True)

    feast_df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print("\nFeast data created successfully.")
    print("Rows:", len(feast_df))
    print("Columns:", len(feast_df.columns))
    print("\nSaved:")
    print(OUTPUT_PATH)

    print("\nColumns:")
    for column in feast_df.columns:
        print("✓", column)


if __name__ == "__main__":
    main()