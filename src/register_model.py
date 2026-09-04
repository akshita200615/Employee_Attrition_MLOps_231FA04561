import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Employee_Attrition_Experiment_231FA04561"
REGISTERED_MODEL_NAME = "EmployeeAttritionClassifier_231FA04561"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def main():

    print("=" * 60)
    print("PHASE 11 - MLflow MODEL REGISTRY")
    print("=" * 60)

    # Find experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        print("\nExperiment not found.")
        print("Please run the training pipeline first.")
        return

    print("\nExperiment found:")
    print(EXPERIMENT_NAME)

    # Find Random Forest run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'Random Forest'",
        order_by=["metrics.f1_score DESC"]
    )

    if not runs:
        print("\nRandom Forest run not found.")
        print("Please run src/train.py first.")
        return

    best_run = runs[0]

    run_id = best_run.info.run_id
    f1_score = best_run.data.metrics.get("f1_score")
    accuracy = best_run.data.metrics.get("accuracy")
    roc_auc = best_run.data.metrics.get("roc_auc")

    print("\nSelected model:")
    print("Model       : Random Forest")
    print("Run ID      :", run_id)
    print("F1 Score    :", f1_score)
    print("Accuracy    :", accuracy)
    print("ROC-AUC     :", roc_auc)

    # Model artifact URI
    model_uri = f"runs:/{run_id}/model"

    print("\nRegistering model...")
    print("Model URI:", model_uri)

    result = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    print("\n" + "=" * 60)
    print("MODEL REGISTERED SUCCESSFULLY")
    print("=" * 60)

    print("\nRegistered Model:")
    print(REGISTERED_MODEL_NAME)

    print("Version      :", result.version)
    print("Run ID       :", run_id)
    print("F1 Score     :", f1_score)

    # Add useful tags to the model version
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=result.version,
        key="model_type",
        value="Random Forest"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=result.version,
        key="f1_score",
        value=str(f1_score)
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=result.version,
        key="purpose",
        value="Baseline model for employee attrition prediction"
    )

    print("\nModel version tags added successfully.")

    print("\nYou can now view the model in MLflow UI.")
    print("http://127.0.0.1:5000")


if __name__ == "__main__":
    main()