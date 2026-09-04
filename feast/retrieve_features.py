import pandas as pd
from feast import FeatureStore


def main():

    print("=" * 60)
    print("FEAST FEATURE RETRIEVAL")
    print("=" * 60)

    store = FeatureStore(
        repo_path="feast"
    )

    # Load Feast source data
    source_data = pd.read_parquet(
        "feast/data/employee_features.parquet"
    )

    # Select a few employees for demonstration
    entity_df = source_data[
        [
            "EmployeeNumber",
            "event_timestamp"
        ]
    ].head(10)

    print("\nRequesting features from Feast...")

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "employee_features:Age",
            "employee_features:JobSatisfaction",
            "employee_features:TotalExperience",
            "employee_features:PromotionRatio",
            "employee_features:JobSatisfactionScore",
            "employee_features:YearsPerCompanyAge",
            "employee_features:CareerStagnationIndex"
        ]
    ).to_df()

    print("\nFeatures retrieved successfully!")

    print("\nRetrieved data:")
    print(training_df)

    print("\nShape:")
    print(training_df.shape)

    print("\nFeature retrieval completed successfully.")


if __name__ == "__main__":
    main()