import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "EmployeeAttritionClassifier_231FA04561"
CHAMPION_VERSION = "1"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient(
    tracking_uri=MLFLOW_TRACKING_URI
)


def main():

    print("=" * 60)
    print("PHASE 13 - CHAMPION MODEL SELECTION")
    print("=" * 60)

    print("\nRegistered Model:")
    print(REGISTERED_MODEL_NAME)

    # Get Version 1
    version = client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version=CHAMPION_VERSION
    )

    print("\nSelected Champion:")
    print("Version     :", version.version)
    print("Run ID      :", version.run_id)

    # Get run metrics
    run = client.get_run(version.run_id)

    model_name = run.data.tags.get("mlflow.runName")
    f1_score = run.data.metrics.get("f1_score")
    accuracy = run.data.metrics.get("accuracy")
    precision = run.data.metrics.get("precision")
    recall = run.data.metrics.get("recall")
    roc_auc = run.data.metrics.get("roc_auc")

    print("Model       :", model_name)
    print("Accuracy    :", accuracy)
    print("Precision   :", precision)
    print("Recall      :", recall)
    print("F1 Score    :", f1_score)
    print("ROC-AUC     :", roc_auc)

    # Assign Champion alias
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "champion",
        CHAMPION_VERSION
    )

    # Add model version tags
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=CHAMPION_VERSION,
        key="status",
        value="champion"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=CHAMPION_VERSION,
        key="selection_metric",
        value="F1 Score"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=CHAMPION_VERSION,
        key="selection_reason",
        value="Highest F1 score among registered models"
    )

    print("\n" + "=" * 60)
    print("CHAMPION MODEL SET SUCCESSFULLY")
    print("=" * 60)

    print("\nChampion Model:")
    print(REGISTERED_MODEL_NAME)

    print("Champion Version : 1")
    print("Champion Alias   : champion")
    print("Model            :", model_name)
    print("F1 Score         :", f1_score)

    print("\nReason:")
    print("Version 1 achieved the highest F1 score.")

    print("\nMLflow UI:")
    print("http://127.0.0.1:5000")


if __name__ == "__main__":
    main()