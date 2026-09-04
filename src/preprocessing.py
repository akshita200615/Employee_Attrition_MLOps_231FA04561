import os
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_PATH = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
PROCESSED_DIR = "data/processed"


def load_data():
    return pd.read_csv(RAW_PATH)


def preprocess_data(df):
    df = df.copy()

    # Remove columns that contain no useful predictive information
    columns_to_drop = [
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours"
    ]

    existing_columns = [
        column for column in columns_to_drop
        if column in df.columns
    ]

    df = df.drop(columns=existing_columns)

    # Convert target to numeric
    df["Attrition"] = df["Attrition"].map({
        "Yes": 1,
        "No": 0
    })

    return df


def split_data(df):

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def save_data(X_train, X_test, y_train, y_test):

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train = X_train.copy()
    train["Attrition"] = y_train.values

    test = X_test.copy()
    test["Attrition"] = y_test.values

    train.to_csv(
        os.path.join(PROCESSED_DIR, "train.csv"),
        index=False
    )

    test.to_csv(
        os.path.join(PROCESSED_DIR, "test.csv"),
        index=False
    )


def main():

    print("=" * 60)
    print("DVC STAGE 1 - DATA PREPROCESSING")
    print("=" * 60)

    print("\nLoading dataset...")

    df = load_data()

    print("Original shape:", df.shape)

    print("\nApplying preprocessing...")

    df = preprocess_data(df)

    print("After preprocessing:", df.shape)

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = split_data(df)

    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))

    print("\nSaving processed datasets...")

    save_data(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nPreprocessing completed successfully.")

    print("\nCreated:")
    print("✓ data/processed/train.csv")
    print("✓ data/processed/test.csv")


if __name__ == "__main__":
    main()