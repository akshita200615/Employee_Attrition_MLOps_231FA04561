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


EXPERIMENT_NAME = "Employee_Attrition_Experiment_231FA04561"


def main():

    X_train = joblib.load(
        "data/processed/X_train.pkl"
    )

    X_test = joblib.load(
        "data/processed/X_test.pkl"
    )

    y_train = joblib.load(
        "data/processed/y_train.pkl"
    )

    y_test = joblib.load(
        "data/processed/y_test.pkl"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    model = RandomForestClassifier(
        random_state=42
    )

    parameter_grid = {

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

    print("Starting hyperparameter tuning...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    best_model = grid_search.best_estimator_

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

    with mlflow.start_run(
        run_name="Tuned_Random_Forest"
    ):

        mlflow.log_params(
            grid_search.best_params_
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
            best_model,
            "model"
        )

    joblib.dump(
        best_model,
        "models/tuned_random_forest.pkl"
    )

    print("\nBest parameters:")
    print(
        grid_search.best_params_
    )

    print("\nResults:")

    print(
        "Accuracy:",
        round(accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall:",
        round(recall, 4)
    )

    print(
        "F1:",
        round(f1, 4)
    )

    print(
        "ROC-AUC:",
        round(roc_auc, 4)
    )

    print("\nTuning completed.")


if __name__ == "__main__":
    main()