import os
import joblib
import mlflow
import mlflow.sklearn

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


EXPERIMENT_NAME = "Employee_Attrition_Experiment_231FA04561"


def evaluate_model(model, X_test, y_test):

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


def train_model(
    model,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test
):

    with mlflow.start_run(
        run_name=model_name
    ):

        model.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        mlflow.log_params(
            model.get_params()
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        print("\n==========================")
        print(model_name)
        print("==========================")

        for key, value in metrics.items():
            print(
                f"{key}: {value:.4f}"
            )

        return model, metrics


def main():

    os.makedirs(
        "models",
        exist_ok=True
    )

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

    # -----------------------------
    # Experiment 1
    # -----------------------------

    logistic = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    train_model(
        logistic,
        "Logistic_Regression",
        X_train,
        y_train,
        X_test,
        y_test
    )

    # -----------------------------
    # Experiment 2
    # -----------------------------

    tree = DecisionTreeClassifier(
        random_state=42
    )

    train_model(
        tree,
        "Decision_Tree",
        X_train,
        y_train,
        X_test,
        y_test
    )

    # -----------------------------
    # Experiment 3
    # -----------------------------

    forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model, metrics = train_model(
        forest,
        "Random_Forest",
        X_train,
        y_train,
        X_test,
        y_test
    )

    joblib.dump(
        model,
        "models/random_forest_baseline.pkl"
    )

    print("\nBaseline training completed.")


if __name__ == "__main__":
    main()