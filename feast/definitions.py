from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64


# ---------------------------------------------------------
# Data source
# ---------------------------------------------------------

employee_source = FileSource(
    name="employee_features_source",
    path="data/employee_features.parquet",
    timestamp_field="event_timestamp"
)


# ---------------------------------------------------------
# Entity
# ---------------------------------------------------------

employee = Entity(
    name="employee",
    join_keys=["EmployeeNumber"],
    description="IBM HR employee identifier"
)


# ---------------------------------------------------------
# Feature View
# ---------------------------------------------------------

employee_features = FeatureView(
    name="employee_features",
    entities=[employee],
    ttl=timedelta(days=3650),
    schema=[
        Field(
            name="Age",
            dtype=Int64
        ),
        Field(
            name="JobSatisfaction",
            dtype=Int64
        ),
        Field(
            name="EnvironmentSatisfaction",
            dtype=Int64
        ),
        Field(
            name="RelationshipSatisfaction",
            dtype=Int64
        ),
        Field(
            name="YearsAtCompany",
            dtype=Int64
        ),
        Field(
            name="YearsInCurrentRole",
            dtype=Int64
        ),
        Field(
            name="YearsWithCurrManager",
            dtype=Int64
        ),
        Field(
            name="YearsSinceLastPromotion",
            dtype=Int64
        ),
        Field(
            name="TotalExperience",
            dtype=Float64
        ),
        Field(
            name="PromotionRatio",
            dtype=Float64
        ),
        Field(
            name="JobSatisfactionScore",
            dtype=Float64
        ),
        Field(
            name="YearsPerCompanyAge",
            dtype=Float64
        ),
        Field(
            name="CareerStagnationIndex",
            dtype=Float64
        )
    ],
    source=employee_source,
    online=True
)