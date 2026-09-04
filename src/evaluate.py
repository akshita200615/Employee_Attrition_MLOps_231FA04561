import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

X_TEST_PATH = "data/processed/X_test.pkl"
Y_TEST_PATH = "data/processed/y_test.pkl"

MODEL_DIR = "models"
REPORT_DIR = "reports"

RESULT_PATH = "reports/model_comparison.csv"


# ---------------------------------------------------------
# Load test data
# ---------------------------------------------------------

def load_test_data():

    X_test = joblib.load(
        X_TEST_PATH
    )

    y_test = joblib.load(
        Y_TEST_PATH
    )

    return X_test, y_test


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    results = {
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

    return results, predictions, probabilities


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    X_test, y_test = load_test_data()

    # -----------------------------------------------------
    # Models
    # -----------------------------------------------------

    model_files = {
        "Logistic Regression":
            "logistic_regression.pkl",

        "Decision Tree":
            "decision_tree.pkl",

        "Random Forest":
            "random_forest.pkl",

        "Tuned Random Forest":
            "tuned_random_forest.pkl"
    }

    all_results = []

    predictions_dict = {}
    probabilities_dict = {}

    # -----------------------------------------------------
    # Evaluate each model
    # -----------------------------------------------------

    for model_name, filename in model_files.items():

        model_path = os.path.join(
            MODEL_DIR,
            filename
        )

        print("\nEvaluating:")
        print(model_name)

        model = joblib.load(
            model_path
        )

        results, predictions, probabilities = evaluate_model(
            model,
            X_test,
            y_test
        )

        results["model"] = model_name

        all_results.append(
            results
        )

        predictions_dict[model_name] = predictions

        probabilities_dict[model_name] = probabilities

        print(
            f"Accuracy : {results['accuracy']:.4f}"
        )

        print(
            f"Precision: {results['precision']:.4f}"
        )

        print(
            f"Recall   : {results['recall']:.4f}"
        )

        print(
            f"F1 Score : {results['f1_score']:.4f}"
        )

        print(
            f"ROC-AUC  : {results['roc_auc']:.4f}"
        )

    # -----------------------------------------------------
    # Create comparison table
    # -----------------------------------------------------

    comparison = pd.DataFrame(
        all_results
    )

    comparison = comparison[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc"
        ]
    ]

    comparison = comparison.sort_values(
        by="f1_score",
        ascending=False
    )

    comparison.to_csv(
        RESULT_PATH,
        index=False
    )

    # -----------------------------------------------------
    # Print comparison
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        comparison.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Best model
    # -----------------------------------------------------

    best_model = comparison.iloc[0]["model"]

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(
        "Best model based on F1 score:",
        best_model
    )

    # -----------------------------------------------------
    # Confusion Matrix - Best Model
    # -----------------------------------------------------

    best_predictions = predictions_dict[
        best_model
    ]

    cm = confusion_matrix(
        y_test,
        best_predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    display.plot()

    plt.title(
        f"Confusion Matrix - {best_model}"
    )

    plt.tight_layout()

    plt.savefig(
        "reports/confusion_matrix.png"
    )

    plt.close()

    # -----------------------------------------------------
    # ROC Curves
    # -----------------------------------------------------

    plt.figure()

    for model_name, probabilities in probabilities_dict.items():

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC={auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve - Employee Attrition Models"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "reports/roc_curve.png"
    )

    plt.close()

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\nReports created:")

    print(
        "✓ reports/model_comparison.csv"
    )

    print(
        "✓ reports/confusion_matrix.png"
    )

    print(
        "✓ reports/roc_curve.png"
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
