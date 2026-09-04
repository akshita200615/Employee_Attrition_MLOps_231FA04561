import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate(
    name,
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    result = {

        "Model": name,

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_test,
            probabilities
        )
    }

    print("\n", name)

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return result


def main():

    X_test = joblib.load(
        "data/processed/X_test.pkl"
    )

    y_test = joblib.load(
        "data/processed/y_test.pkl"
    )

    baseline_rf = joblib.load(
        "models/random_forest_baseline.pkl"
    )

    tuned_rf = joblib.load(
        "models/tuned_random_forest.pkl"
    )

    results = []

    results.append(
        evaluate(
            "Random Forest",
            baseline_rf,
            X_test,
            y_test
        )
    )

    results.append(
        evaluate(
            "Tuned Random Forest",
            tuned_rf,
            X_test,
            y_test
        )
    )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        "reports/model_comparison.csv",
        index=False
    )

    print("\n============================")
    print("MODEL COMPARISON")
    print("============================")

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved: reports/model_comparison.csv"
    )


if __name__ == "__main__":
    main()