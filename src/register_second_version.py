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
            "start_time DESC"
        ]
    )

    for run in runs:

        if run.data.tags.get(
            "mlflow.runName"
        ) == "Tuned_Random_Forest":

            run_id = run.info.run_id

            model_uri = (
                f"runs:/{run_id}/model"
            )

            registered = mlflow.register_model(
                model_uri=model_uri,
                name=MODEL_NAME
            )

            print(
                "Second version registered:",
                registered.version
            )

            return

    print(
        "Tuned Random Forest run not found."
    )


if __name__ == "__main__":
    main()