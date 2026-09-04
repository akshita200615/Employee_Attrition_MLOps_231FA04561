import os
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_PATH = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
PROCESSED_DIR = "data/processed"


def load_data():
    """Load the original IBM HR Employee Attrition dataset."""
    df = pd.read_csv(RAW_PATH)
    return df


def preprocess_data(df):
    """Basic preprocessing and target conversion."""

    df = df.copy()

    # EmployeeNumber is only an identifier and is not useful for prediction
    if "EmployeeNumber" in df.columns:
        df = df.drop(columns=["EmployeeNumber"])

    # EmployeeCount has only one unique value
    if "EmployeeCount" in df.columns:
        df = df.drop(columns=["EmployeeCount"])

    # Over18 has only one value
    if "Over18" in df.columns:
        df = df.drop(columns=["Over18"])

    # StandardHours has only one value
    if "StandardHours" in df.columns:
        df = df.drop(columns=["StandardHours"])

    # Convert target variable
    df["Attrition"] = df["Attrition"].map({
        "Yes": 1,
        "No": 0
    })

    return df


def split_data(df):
    """Split dataset into training and testing data."""

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

    print("Loading dataset...")

    df = load_data()

    print("Original shape:", df.shape)

    df = preprocess_data(df)

    print("After preprocessing:", df.shape)

    X_train, X_test, y_train, y_test = split_data(df)

    save_data(X_train, X_test, y_train, y_test)

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nPreprocessing completed successfully.")
    print("Files created:")
    print("data/processed/train.csv")
    print("data/processed/test.csv")


if __name__ == "__main__":
    main()