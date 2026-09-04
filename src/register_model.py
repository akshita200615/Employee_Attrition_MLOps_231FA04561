import mlflow
from mlflow.tracking import MlflowClient


MODEL_NAME = "EmployeeAttritionClassifier_231FA04561"


def main():

    client = MlflowClient()

    experiment = client.get_experiment_by_name(
        "Employee_Attrition_Experiment_231FA04561"
    )

    runs = client.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ],
        order_by=[
            "metrics.f1_score DESC"
        ]
    )

    if len(runs) == 0:
        print("No MLflow runs found.")
        return

    best_run = runs[0]

    run_id = best_run.info.run_id

    model_uri = f"runs:/{run_id}/model"

    print("Best run:", run_id)

    print(
        "F1:",
        best_run.data.metrics.get(
            "f1_score"
        )
    )

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print("\nRegistered model:")
    print(MODEL_NAME)

    print(
        "Version:",
        registered_model.version
    )


if __name__ == "__main__":
    main()