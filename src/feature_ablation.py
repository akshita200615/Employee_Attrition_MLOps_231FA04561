import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

EXPERIMENT_NAME = "Employee_Attrition_Experiment_231FA04561"

TRAIN_ORIGINAL = "data/processed/train.csv"
TEST_ORIGINAL = "data/processed/test.csv"

TRAIN_ENGINEERED = "data/processed/train_features.csv"
TEST_ENGINEERED = "data/processed/test_features.csv"

REPORT_PATH = "reports/feature_ablation.csv"


ENGINEERED_FEATURES = [
    "TotalExperience",
    "PromotionRatio",
    "JobSatisfactionScore",
    "YearsPerCompanyAge",
    "CareerStagnationIndex"
]


# ---------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ---------------------------------------------------------
# Build preprocessing pipeline
# ---------------------------------------------------------

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

    return ColumnTransformer(
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


# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------

def prepare_data(train, test):

    X_train = train.drop(
        columns=["Attrition"]
    )

    y_train = train["Attrition"]

    X_test = test.drop(
        columns=["Attrition"]
    )

    y_test = test["Attrition"]

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# Run one ablation experiment
# ---------------------------------------------------------

def run_experiment(
    experiment_name,
    train,
    test
):

    print("\n" + "=" * 60)
    print(experiment_name)
    print("=" * 60)

    X_train, X_test, y_train, y_test = prepare_data(
        train,
        test
    )

    # Build preprocessing separately for this experiment
    preprocessor = build_preprocessor(
        X_train
    )

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )

    print(
        "Original feature count:",
        X_train.shape[1]
    )

    print(
        "Transformed feature count:",
        X_train_transformed.shape[1]
    )

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")

    model.fit(
        X_train_transformed,
        y_train
    )

    predictions = model.predict(
        X_test_transformed
    )

    probabilities = model.predict_proba(
        X_test_transformed
    )[:, 1]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\nResults:")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    # -----------------------------------------------------
    # MLflow
    # -----------------------------------------------------

    with mlflow.start_run(
        run_name=experiment_name
    ):

        mlflow.log_param(
            "feature_set",
            experiment_name
        )

        mlflow.log_param(
            "n_estimators",
            200
        )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        mlflow.sklearn.log_model(
            model,
            "model"
        )

    return {
        "experiment": experiment_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("FEATURE ABLATION EXPERIMENT")
    print("=" * 60)

    os.makedirs(
        "reports",
        exist_ok=True
    )

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------

    original_train = pd.read_csv(
        TRAIN_ORIGINAL
    )

    original_test = pd.read_csv(
        TEST_ORIGINAL
    )

    engineered_train = pd.read_csv(
        TRAIN_ENGINEERED
    )

    engineered_test = pd.read_csv(
        TEST_ENGINEERED
    )

    # -----------------------------------------------------
    # Experiment 1 - Original features
    # -----------------------------------------------------

    original_results = run_experiment(
        "Ablation - Original Features Only",
        original_train,
        original_test
    )

    # -----------------------------------------------------
    # Experiment 2 - Original + engineered
    # -----------------------------------------------------

    engineered_results = run_experiment(
        "Ablation - Original + Engineered Features",
        engineered_train,
        engineered_test
    )

    # -----------------------------------------------------
    # Comparison
    # -----------------------------------------------------

    comparison = pd.DataFrame(
        [
            original_results,
            engineered_results
        ]
    )

    comparison.to_csv(
        REPORT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("FEATURE ABLATION COMPARISON")
    print("=" * 60)

    print(
        comparison.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Improvement
    # -----------------------------------------------------

    original_f1 = original_results[
        "f1_score"
    ]

    engineered_f1 = engineered_results[
        "f1_score"
    ]

    improvement = (
        engineered_f1 - original_f1
    )

    improvement_percentage = (
        improvement / original_f1
    ) * 100

    print("\n" + "=" * 60)
    print("ENGINEERED FEATURE IMPACT")
    print("=" * 60)

    print(
        f"Original F1 Score  : {original_f1:.4f}"
    )

    print(
        f"Engineered F1 Score: {engineered_f1:.4f}"
    )

    print(
        f"F1 Improvement     : {improvement:.4f}"
    )

    print(
        f"Percentage Change  : {improvement_percentage:.2f}%"
    )

    if improvement > 0:

        print(
            "\nConclusion: Engineered features improved "
            "the Random Forest F1 score."
        )

    elif improvement < 0:

        print(
            "\nConclusion: Engineered features reduced "
            "the Random Forest F1 score."
        )

    else:

        print(
            "\nConclusion: Engineered features produced "
            "no change in F1 score."
        )

    print("\nSaved:")
    print(REPORT_PATH)

    print("\nFeature ablation completed successfully.")


if __name__ == "__main__":
    main()