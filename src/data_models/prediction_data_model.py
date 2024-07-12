import pandas as pd
from pydantic import BaseModel, validator

from schema.data_schema import TimeStepClassificationSchema


class PredictionsValidationError(Exception):
    pass


def get_predictions_validator(schema: TimeStepClassificationSchema) -> BaseModel:
    """
    Returns a dynamic Pydantic data validator class based on the provided schema.

    The resulting validator checks the following:

    1. That the input DataFrame is not empty.
    2. That the input DataFrame contains at least two rows.
    3. That the input DataFrame contains the ID field specified in the schema.
    4. That the input DataFrame contains the time field specified in the schema.
    5. That the input DataFrame contains all fields named as target classes.
    6. That the predicted probability columns contain values between 0 and 1.
    7. That the sum of predicted probabilities are between 0.999 and 1.001.


    If any of these checks fail, the validator will raise a ValueError.

    Args:
        schema (TimeStepClassificationSchema): An instance of TimeStepClassificationSchema.
        prediction_field_name (str): The name of the column containing the predictions.

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
                raise PredictionsValidationError(
                    "PredictionsValidationError: The provided predictions file is empty. "
                    "No scores can be generated. "
                ) 
            
            # Check if DataFrame has at least 2 rows
            if len(data) < 2:
                raise PredictionsValidationError(
                    "Malformed predictions file. The provided predictions file must have at least 2 rows."
                    f"Given'{len(data)}' rows"
                )

            if schema.id_col not in data.columns:
                raise PredictionsValidationError(
                    "PredictionsValidationError: Malformed predictions file. "
                    f"ID field '{schema.id}' is not present in predictions file."
                )

            if schema.time_col not in data.columns:
                raise PredictionsValidationError(
                    "PredictionsValidationError: Malformed predictions file. "
                    f"Time field '{schema.time_col}' is not present in predictions file."
                )

            data.columns = [str(c) for c in data.columns]
            target_classes = [str(c) for c in schema.target_classes]
            missing_classes = set(target_classes) - set(data.columns)
            if missing_classes:
                raise PredictionsValidationError(
                    "Malformed predictions file. Target field(s) "
                    f"{missing_classes} missing in predictions file.\n"
                    "Please ensure that the predictions file contains "
                    f"columns named {schema.target_classes} representing "
                    "predicted class probabilities"
                )

            # Check if probabilities are valid
            for class_ in target_classes:
                if not data[class_].between(0, 1).all():
                    raise PredictionsValidationError(
                        f"Some values in the '{class_}' column are not valid "
                        "probabilities. All probabilities should be numbers between "
                        "0 and 1, inclusive."
                    )

            # Check if sum of probabilities are close to 1 for each row
            probability_sums = data[schema.target_classes].sum(axis=1)
            if not ((probability_sums >= 0.999) & (probability_sums <= 1.001)).all():
                rows_with_errors = \
                    probability_sums[~((probability_sums >= 0.999) & \
                        (probability_sums <= 1.001))].index.tolist()
                raise PredictionsValidationError(
                    f"Sum of predicted probabilities for rows {rows_with_errors} "
                    "are not close to 1. Ensure that the sum of probabilities for "
                    "each row is between 0.999 and 1.001."
                )

            return data
    return DataValidator


def validate_predictions(
    predictions: pd.DataFrame,
    data_schema: TimeStepClassificationSchema
) -> pd.DataFrame:
    """
    Validates the predictions using the provided schema.

    Args:
        predictions (pd.DataFrame): Predictions data to validate.
        data_schema (TimeStepClassificationSchema): An instance of TimeStepClassificationSchema.

    Returns:
        pd.DataFrame: The validated data.
    """
    DataValidator = get_predictions_validator(data_schema)
    try:
        validated_data = DataValidator(data=predictions)
        return validated_data.data
    except ValueError as exc:
        raise ValueError(f"Prediction data validation failed: {str(exc)}") from exc