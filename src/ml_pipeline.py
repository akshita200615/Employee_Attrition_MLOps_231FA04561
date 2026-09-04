import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


TARGET = "Attrition"

TRAIN_PATH = "data/processed/train_features.csv"
TEST_PATH = "data/processed/test_features.csv"

PREPROCESSOR_PATH = "models/preprocessor.pkl"


def load_data():

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    return train, test


def build_preprocessor(X):

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    return preprocessor


def main():

    print("Loading feature-engineered data...")

    train, test = load_data()

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]

    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    print("\nBuilding preprocessing pipeline...")

    preprocessor = build_preprocessor(X_train)

    X_train_transformed = preprocessor.fit_transform(X_train)

    X_test_transformed = preprocessor.transform(X_test)

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    joblib.dump(
        X_train_transformed,
        "data/processed/X_train.pkl"
    )

    joblib.dump(
        X_test_transformed,
        "data/processed/X_test.pkl"
    )

    joblib.dump(
        y_train,
        "data/processed/y_train.pkl"
    )

    joblib.dump(
        y_test,
        "data/processed/y_test.pkl"
    )

    print("\nPreprocessing pipeline completed.")

    print(
        "Transformed training shape:",
        X_train_transformed.shape
    )

    print(
        "Transformed testing shape:",
        X_test_transformed.shape
    )

    print("\nSaved:")
    print("models/preprocessor.pkl")
    print("data/processed/X_train.pkl")
    print("data/processed/X_test.pkl")
    print("data/processed/y_train.pkl")
    print("data/processed/y_test.pkl")


if __name__ == "__main__":
    main()