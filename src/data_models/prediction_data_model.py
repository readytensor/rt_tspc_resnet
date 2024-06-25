import pandas as pd
from pydantic import BaseModel, validator

from schema.data_schema import TSAnnotationSchema


def get_predictions_validator(
    schema: TSAnnotationSchema, test_data_length: int
) -> BaseModel:
    """
    Returns a dynamic Pydantic data validator class based on the provided schema.

    The resulting validator checks the following:

    1. That the input DataFrame is not empty.
    2. That the input DataFrame contains the ID field specified in the schema.
    3. That the input DataFrame contains the time field specified in the schema.
    4. That the length of the input DataFrame is equal to the length of the test data.



    If any of these checks fail, the validator will raise a ValueError.

    Args:
        schema (TSAnnotationSchema): An instance of TSAnnotationSchema.
        prediction_field_name (str): The name of the column containing the predictions.
        test_data_length (int): The length of the test data.

    Returns:
        BaseModel: A dynamic Pydantic BaseModel class for data validation.
    """

    class DataValidator(BaseModel):
        data: pd.DataFrame

        class Config:
            arbitrary_types_allowed = True

        @validator("data", allow_reuse=True)
        def validate_dataframe(cls, data):

            # Check if DataFrame is empty
            if data.empty:
                raise ValueError(
                    "ValueError: The provided predictions file is empty. "
                    "No scores can be generated. "
                )

            if schema.id_col not in data.columns:
                raise ValueError(
                    "ValueError: Malformed predictions file. "
                    f"ID field '{schema.id}' is not present in predictions file."
                )

            if schema.time_col not in data.columns:
                raise ValueError(
                    "ValueError: Malformed predictions file. "
                    f"Time field '{schema.time_col}' is not present in predictions file."
                )

            if len(data) != test_data_length:
                raise ValueError(
                    "ValueError: Malformed predictions file. "
                    f"Length of predictions file is not equal to the length of the test data."
                )
            return data

    return DataValidator


def validate_predictions(
    predictions: pd.DataFrame,
    data_schema: TSAnnotationSchema,
    test_data_length: int,
) -> pd.DataFrame:
    """
    Validates the predictions using the provided schema.

    Args:
        predictions (pd.DataFrame): Predictions data to validate.
        data_schema (TSAnnotationSchema): An instance of TSAnnotationSchema.

    Returns:
        pd.DataFrame: The validated data.
    """
    DataValidator = get_predictions_validator(data_schema, test_data_length)
    try:
        validated_data = DataValidator(data=predictions)
        return validated_data.data
    except ValueError as exc:
        raise ValueError(f"Prediction data validation failed: {str(exc)}") from exc


if __name__ == "__main__":
    schema_dict = {
        "title": "test dataset",
        "description": "test dataset",
        "modelCategory": "multiclass_classification",
        "schemaVersion": 1.0,
        "inputDataFormat": "CSV",
        "id": {"name": "id", "description": "unique identifier."},
        "target": {
            "name": "target_field",
            "description": "some target desc.",
            "classes": ["A", "B", "C"],
        },
        "features": [
            {
                "name": "numeric_feature_1",
                "description": "some desc.",
                "dataType": "NUMERIC",
                "example": 50,
                "nullable": True,
            },
            {
                "name": "categorical_feature_1",
                "description": "some desc.",
                "dataType": "CATEGORICAL",
                "categories": ["X", "Y", "Z"],
                "nullable": True,
            },
        ],
    }
    schema_provider = TSAnnotationSchema(schema_dict)
    predictions = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "A": [0.9, 0.2, 0.8, 0.1, 0.85],
            "B": [0.05, 0.5, 0.1, 0.5, 0.05],
            "C": [0.05, 0.3, 0.1, 0.4, 0.1],
        }
    )

    validated_data = validate_predictions(predictions, schema_provider)
