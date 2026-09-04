import os
import pandas as pd


TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

TRAIN_OUTPUT = "data/processed/train_features.csv"
TEST_OUTPUT = "data/processed/test_features.csv"


ENGINEERED_FEATURES = [
    "TotalExperience",
    "PromotionRatio",
    "JobSatisfactionScore",
    "YearsPerCompanyAge",
    "CareerStagnationIndex"
]


def create_features(df):

    df = df.copy()

    # ---------------------------------------------------------
    # 1. Total Experience
    # ---------------------------------------------------------

    df["TotalExperience"] = (
        df["YearsAtCompany"]
        + df["YearsInCurrentRole"]
        + df["YearsWithCurrManager"]
    )

    # ---------------------------------------------------------
    # 2. Promotion Ratio
    # ---------------------------------------------------------

    df["PromotionRatio"] = (
        df["YearsSinceLastPromotion"]
        / (df["YearsAtCompany"] + 1)
    )

    # ---------------------------------------------------------
    # 3. Job Satisfaction Score
    # ---------------------------------------------------------

    df["JobSatisfactionScore"] = (
        df["JobSatisfaction"]
        + df["EnvironmentSatisfaction"]
        + df["RelationshipSatisfaction"]
    ) / 3

    # ---------------------------------------------------------
    # 4. Years Per Company Age
    # ---------------------------------------------------------

    df["YearsPerCompanyAge"] = (
        df["YearsAtCompany"]
        / (df["Age"] + 1)
    )

    # ---------------------------------------------------------
    # 5. Career Stagnation Index
    # ---------------------------------------------------------

    df["CareerStagnationIndex"] = (
        df["YearsAtCompany"]
        / (df["YearsSinceLastPromotion"] + 1)
    )

    return df


def main():

    print("=" * 60)
    print("DVC STAGE 2 - FEATURE ENGINEERING")
    print("=" * 60)

    print("\nLoading processed datasets...")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print("Training shape:", train.shape)
    print("Testing shape :", test.shape)

    print("\nCreating engineered features...")

    train_features = create_features(train)
    test_features = create_features(test)

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    train_features.to_csv(
        TRAIN_OUTPUT,
        index=False
    )

    test_features.to_csv(
        TEST_OUTPUT,
        index=False
    )

    print("\nEngineered features created:")

    for feature in ENGINEERED_FEATURES:
        print("✓", feature)

    print("\nTraining feature shape:", train_features.shape)
    print("Testing feature shape :", test_features.shape)

    print("\nFeature engineering completed successfully.")

    print("\nCreated:")
    print("✓", TRAIN_OUTPUT)
    print("✓", TEST_OUTPUT)


if __name__ == "__main__":
    main()