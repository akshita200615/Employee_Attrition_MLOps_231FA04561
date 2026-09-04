import pandas as pd
import os


RAW_PATH = "../data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"

OUTPUT_PATH = "data/employee_features.csv"


def main():

    df = pd.read_csv(RAW_PATH)

    df["TotalExperience"] = (
        df["YearsAtCompany"]
        + df["YearsInCurrentRole"]
        + df["YearsWithCurrManager"]
    )

    df["PromotionRatio"] = (
        df["YearsSinceLastPromotion"]
        / (df["YearsAtCompany"] + 1)
    )

    df["JobSatisfactionScore"] = (
        df["JobSatisfaction"]
        + df["EnvironmentSatisfaction"]
        + df["RelationshipSatisfaction"]
    ) / 3

    df["YearsPerCompanyAge"] = (
        df["YearsAtCompany"]
        / (df["Age"] + 1)
    )

    df["CareerStagnationIndex"] = (
        df["YearsAtCompany"]
        / (df["YearsSinceLastPromotion"] + 1)
    )

    feast_columns = [
        "EmployeeNumber",
        "Age",
        "YearsAtCompany",
        "YearsSinceLastPromotion",
        "TotalExperience",
        "PromotionRatio",
        "JobSatisfactionScore",
        "YearsPerCompanyAge",
        "CareerStagnationIndex"
    ]

    feast_df = df[feast_columns].copy()

    feast_df["event_timestamp"] = pd.Timestamp.now()

    os.makedirs("data", exist_ok=True)

    feast_df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print("Feast dataset created:")
    print(OUTPUT_PATH)

    print("\nShape:")
    print(feast_df.shape)

    print("\nColumns:")
    print(feast_df.columns.tolist())


if __name__ == "__main__":
    main()