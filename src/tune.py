import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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

MODEL_PATH = "models/tuned_random_forest.pkl"


# ---------------------------------------------------------
# MLflow
# ---------------------------------------------------------

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data():

    X_train = joblib.load(
        X_TRAIN_PATH
    )

    X_test = joblib.load(
        X_TEST_PATH
    )

    y_train = joblib.load(
        Y_TRAIN_PATH
    )

    y_test = joblib.load(
        Y_TEST_PATH
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()

    print("\nTraining data:", X_train.shape)
    print("Testing data :", X_test.shape)

    # -----------------------------------------------------
    # Base Random Forest
    # -----------------------------------------------------

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    # -----------------------------------------------------
    # Hyperparameter grid
    # -----------------------------------------------------

    param_grid = {

        "n_estimators": [
            100,
            200
        ],

        "max_depth": [
            5,
            10,
            15
        ],

        "min_samples_split": [
            2,
            5
        ],

        "min_samples_leaf": [
            1,
            2
        ]
    }

    print("\nHyperparameter grid:")

    for parameter, values in param_grid.items():
        print(
            parameter,
            ":",
            values
        )

    # -----------------------------------------------------
    # Grid Search
    # -----------------------------------------------------

    print("\nStarting GridSearchCV...")
    print("Scoring: F1")
    print("Cross-validation: 5 folds")

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # Best model
    # -----------------------------------------------------

    best_model = grid_search.best_estimator_

    print("\n" + "=" * 60)
    print("BEST HYPERPARAMETERS")
    print("=" * 60)

    print(
        grid_search.best_params_
    )

    print(
        "\nBest cross-validation F1:",
        round(
            grid_search.best_score_,
            4
        )
    )

    # -----------------------------------------------------
    # Test evaluation
    # -----------------------------------------------------

    predictions = best_model.predict(
        X_test
    )

    probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

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

    print("\n" + "=" * 60)
    print("TUNED RANDOM FOREST RESULTS")
    print("=" * 60)

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
    # MLflow Run
    # -----------------------------------------------------

    print("\nLogging experiment to MLflow...")

    with mlflow.start_run(
        run_name="Tuned Random Forest"
    ):

        # Log best parameters
        mlflow.log_params(
            grid_search.best_params_
        )

        # Log CV score
        mlflow.log_metric(
            "best_cv_f1",
            grid_search.best_score_
        )

        # Log test metrics
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

        # Log model
        mlflow.sklearn.log_model(
            best_model,
            "model"
        )

        run_id = mlflow.active_run().info.run_id

        print("\nMLflow Run ID:")
        print(run_id)

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    print("\nModel saved:")
    print(MODEL_PATH)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()