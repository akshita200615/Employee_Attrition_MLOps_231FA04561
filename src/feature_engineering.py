import os
import numpy as np
import pandas as pd


TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

TRAIN_OUTPUT = "data/processed/train_features.csv"
TEST_OUTPUT = "data/processed/test_features.csv"


def create_features(df):
    """
    Create meaningful employee-level features.

    Features:
    1. TotalExperience
    2. PromotionRatio
    3. JobSatisfactionScore
    4. YearsPerCompanyAge
    5. CareerStagnationIndex - original feature
    """

    df = df.copy()

    # --------------------------------------------------
    # Feature 1: TotalExperience
    # --------------------------------------------------
    if {
        "YearsInCurrentRole",
        "YearsAtCompany",
        "YearsWithCurrManager"
    }.issubset(df.columns):

        df["TotalExperience"] = (
            df["YearsAtCompany"]
            + df["YearsInCurrentRole"]
            + df["YearsWithCurrManager"]
        )

    # --------------------------------------------------
    # Feature 2: PromotionRatio
    # --------------------------------------------------
    if {
        "YearsSinceLastPromotion",
        "YearsAtCompany"
    }.issubset(df.columns):

        df["PromotionRatio"] = (
            df["YearsSinceLastPromotion"]
            / (df["YearsAtCompany"] + 1)
        )

    # --------------------------------------------------
    # Feature 3: JobSatisfactionScore
    # --------------------------------------------------
    satisfaction_columns = [
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction"
    ]

    if all(column in df.columns for column in satisfaction_columns):

        df["JobSatisfactionScore"] = (
            df["JobSatisfaction"]
            + df["EnvironmentSatisfaction"]
            + df["RelationshipSatisfaction"]
        ) / 3

    # --------------------------------------------------
    # Feature 4: YearsPerCompanyAge
    # --------------------------------------------------
    if {
        "YearsAtCompany",
        "Age"
    }.issubset(df.columns):

        df["YearsPerCompanyAge"] = (
            df["YearsAtCompany"]
            / (df["Age"] + 1)
        )

    # --------------------------------------------------
    # Feature 5: ORIGINAL FEATURE
    # CareerStagnationIndex
    # --------------------------------------------------
    if {
        "YearsAtCompany",
        "YearsSinceLastPromotion"
    }.issubset(df.columns):

        df["CareerStagnationIndex"] = (
            df["YearsAtCompany"]
            / (df["YearsSinceLastPromotion"] + 1)
        )

    return df


def main():

    print("Loading processed datasets...")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print("Creating engineered features...")

    train_features = create_features(train)
    test_features = create_features(test)

    os.makedirs("data/processed", exist_ok=True)

    train_features.to_csv(
        TRAIN_OUTPUT,
        index=False
    )

    test_features.to_csv(
        TEST_OUTPUT,
        index=False
    )

    print("\nFeature engineering completed.")

    print("\nNew features created:")

    features = [
        "TotalExperience",
        "PromotionRatio",
        "JobSatisfactionScore",
        "YearsPerCompanyAge",
        "CareerStagnationIndex"
    ]

    for feature in features:
        if feature in train_features.columns:
            print("✓", feature)

    print("\nTraining feature shape:", train_features.shape)
    print("Testing feature shape:", test_features.shape)

    print("\nFiles created:")
    print(TRAIN_OUTPUT)
    print(TEST_OUTPUT)


if __name__ == "__main__":
    main()