import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

X_TRAIN_PATH = "data/processed/X_train.pkl"
X_TEST_PATH = "data/processed/X_test.pkl"
Y_TRAIN_PATH = "data/processed/y_train.pkl"
Y_TEST_PATH = "data/processed/y_test.pkl"

MODEL_DIR = "models"


# ---------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data():

    X_train = joblib.load(X_TRAIN_PATH)
    X_test = joblib.load(X_TEST_PATH)

    y_train = joblib.load(Y_TRAIN_PATH)
    y_test = joblib.load(Y_TEST_PATH)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

def calculate_metrics(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

    return metrics


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

def train_model(model, model_name, X_train, X_test, y_train, y_test):

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    with mlflow.start_run(
        run_name=model_name
    ):

        # Train
        print("\nTraining model...")

        model.fit(
            X_train,
            y_train
        )

        # Evaluate
        metrics = calculate_metrics(
            model,
            X_test,
            y_test
        )

        # Log parameters
        params = model.get_params()

        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model"
        )

        # Save local model
        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        filename = model_name.lower().replace(
            " ",
            "_"
        ) + ".pkl"

        model_path = os.path.join(
            MODEL_DIR,
            filename
        )

        joblib.dump(
            model,
            model_path
        )

        # Print results
        print("\nResults:")

        for metric, value in metrics.items():

            print(
                f"{metric}: {value:.4f}"
            )

        print("\nSaved model:")
        print(model_path)

        print("\nMLflow Run ID:")
        print(mlflow.active_run().info.run_id)

    return metrics


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("EMPLOYEE ATTRITION MODEL TRAINING")
    print("=" * 60)

    print("\nLoading transformed data...")

    X_train, X_test, y_train, y_test = load_data()

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    print("\nMLflow experiment:")
    print(EXPERIMENT_NAME)

    # -----------------------------------------------------
    # Model 1 - Logistic Regression
    # -----------------------------------------------------

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    train_model(
        logistic_model,
        "Logistic Regression",
        X_train,
        X_test,
        y_train,
        y_test
    )

    # -----------------------------------------------------
    # Model 2 - Decision Tree
    # -----------------------------------------------------

    decision_tree_model = DecisionTreeClassifier(
        random_state=42
    )

    train_model(
        decision_tree_model,
        "Decision Tree",
        X_train,
        X_test,
        y_train,
        y_test
    )

    # -----------------------------------------------------
    # Model 3 - Random Forest
    # -----------------------------------------------------

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    train_model(
        random_forest_model,
        "Random Forest",
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\n" + "=" * 60)
    print("ALL 3 MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()