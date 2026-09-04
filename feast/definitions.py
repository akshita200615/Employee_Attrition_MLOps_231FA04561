from datetime import timedelta

from feast import (
    Entity,
    FeatureView,
    Field
)

from feast.types import Float32, Int64


employee = Entity(
    name="employee",
    join_keys=["EmployeeNumber"]
)


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
            name="YearsAtCompany",
            dtype=Int64
        ),
        Field(
            name="YearsSinceLastPromotion",
            dtype=Int64
        ),
        Field(
            name="TotalExperience",
            dtype=Int64
        ),
        Field(
            name="PromotionRatio",
            dtype=Float32
        ),
        Field(
            name="JobSatisfactionScore",
            dtype=Float32
        ),
        Field(
            name="YearsPerCompanyAge",
            dtype=Float32
        ),
        Field(
            name="CareerStagnationIndex",
            dtype=Float32
        )
    ],
    source=None
)