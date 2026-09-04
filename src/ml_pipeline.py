import os
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

X_TRAIN_PATH = "data/processed/X_train.pkl"
X_TEST_PATH = "data/processed/X_test.pkl"
Y_TRAIN_PATH = "data/processed/y_train.pkl"
Y_TEST_PATH = "data/processed/y_test.pkl"


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    return train, test


def build_preprocessor(X):
    # Find categorical and numerical columns
    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # Combine both pipelines
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

    print("=" * 60)
    print("ML PREPROCESSING PIPELINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    print("\n[1] Loading feature-engineered datasets...")

    train, test = load_data()

    print("Training shape:", train.shape)
    print("Testing shape :", test.shape)

    # ---------------------------------------------------------
    # 2. Separate X and y
    # ---------------------------------------------------------

    print("\n[2] Separating features and target...")

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]

    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape :", y_test.shape)

    # ---------------------------------------------------------
    # 3. Identify columns
    # ---------------------------------------------------------

    categorical_columns = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = X_train.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    print("\nCategorical columns:", len(categorical_columns))
    print("Numerical columns :", len(numerical_columns))

    # ---------------------------------------------------------
    # 4. Build preprocessing pipeline
    # ---------------------------------------------------------

    print("\n[3] Building preprocessing pipeline...")

    preprocessor = build_preprocessor(X_train)

    # ---------------------------------------------------------
    # 5. Fit ONLY on training data
    # ---------------------------------------------------------

    print("\n[4] Fitting preprocessing pipeline on training data...")

    X_train_transformed = preprocessor.fit_transform(X_train)

    # Transform test data using the same fitted pipeline
    X_test_transformed = preprocessor.transform(X_test)

    print("Transformed X_train shape:", X_train_transformed.shape)
    print("Transformed X_test shape :", X_test_transformed.shape)

    # ---------------------------------------------------------
    # 6. Create required directories
    # ---------------------------------------------------------

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # ---------------------------------------------------------
    # 7. Save preprocessing pipeline and transformed data
    # ---------------------------------------------------------

    print("\n[5] Saving preprocessing artifacts...")

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    joblib.dump(
        X_train_transformed,
        X_TRAIN_PATH
    )

    joblib.dump(
        X_test_transformed,
        X_TEST_PATH
    )

    joblib.dump(
        y_train,
        Y_TRAIN_PATH
    )

    joblib.dump(
        y_test,
        Y_TEST_PATH
    )

    # ---------------------------------------------------------
    # 8. Completion message
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nFiles created:")

    print("✓", PREPROCESSOR_PATH)
    print("✓", X_TRAIN_PATH)
    print("✓", X_TEST_PATH)
    print("✓", Y_TRAIN_PATH)
    print("✓", Y_TEST_PATH)


if __name__ == "__main__":
    main()