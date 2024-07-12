import numpy as np
import pandas as pd

from config import paths
from data_models.data_validator import validate_data
from data_models.prediction_data_model import validate_predictions
from logger import get_logger, log_error
from prediction.predictor_model import load_predictor_model, predict_with_model
from preprocessing.preprocess import (
    load_pipeline_of_type,
    transform_data,
)
from schema.data_schema import load_saved_schema
from utils import (
    read_csv_in_directory,
    read_json_as_dict,
    save_dataframe_as_csv,
    cast_time_col,
    ResourceTracker,
)

logger = get_logger(task_name="predict")


def create_predictions_dataframe(
    pred_input: pd.DataFrame,
    predictions_arr: np.ndarray,
    prediction_field_name: str,
    label_encoder: object,
    data_schema: object,
) -> pd.DataFrame:
    """
    Converts the predictions numpy array into a dataframe having the required structure.

    Args:
        pred_input (pd.DataFrame): Test data input.
        predictions_arr (np.ndarray): Forecast from forecasting model.
        prediction_field_name (str): Name of the column containing the predictions.
        label_encoder (object): Label encoder object.
        data_schema (object): Data schema object.

    Returns:
        Predictions as a pandas dataframe
    """
    predictions_df = pred_input.copy()
    label_to_int = label_encoder.encoders[data_schema.target]
    class_names = list(label_to_int.keys())
    predictions_df[class_names] = predictions_arr

    predictions_df = predictions_df[
        [data_schema.id_col, data_schema.time_col] + class_names
    ]
    predictions_df[prediction_field_name] = predictions_df[class_names].idxmax(axis=1)
    return predictions_df


def run_batch_predictions(
    saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
    model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
    test_dir: str = paths.TEST_DIR,
    preprocessing_dir_path: str = paths.PREPROCESSING_DIR_PATH,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    predictions_file_path: str = paths.PREDICTIONS_FILE_PATH,
) -> None:
    """
    Run batch predictions on test data, save the predicted probabilities to a CSV file.

    This function reads test data from the specified directory,
    loads the preprocessing pipeline and pre-trained predictor model,
    transforms the test data using the pipeline,
    makes predictions using the trained predictor model,
    adds ids into the predictions dataframe,
    and saves the predictions as a CSV file.

    Args:
        saved_schema_dir_path (str): Dir path to the saved data schema.
        model_config_file_path (str): Path to the model configuration file.
        test_dir (str): Directory path for the test data.
        preprocessing_dir_path (str): Path to the saved pipeline file.
        predictor_file_path (str): Path to the saved predictor model file.
        predictions_file_path (str): Path where the predictions file will be saved.
    """

    try:
        with ResourceTracker(logger, monitoring_interval=0.1):
            logger.info("Making batch predictions...")

            logger.info("Loading schema...")
            data_schema = load_saved_schema(saved_schema_dir_path)

            logger.info("Loading model config...")
            model_config = read_json_as_dict(model_config_file_path)
            prediction_field_name = model_config["prediction_field_name"]

            # we need the test data to return our final predictions with right columns
            logger.info("Loading test data...")
            test_data = read_csv_in_directory(file_dir_path=test_dir)
            test_data = cast_time_col(
                test_data, data_schema.time_col, data_schema.time_col_dtype
            )
            logger.info("Validating test data...")
            validated_test_data = validate_data(
                data=test_data, data_schema=data_schema, is_train=False
            )

            # fit and transform using pipeline and target encoder, then save them
            logger.info("Loading preprocessing pipeline ...")

            inference_pipeline = load_pipeline_of_type(
                preprocessing_dir_path, pipeline_type="inference"
            )
            transformed_test_data = transform_data(
                inference_pipeline, validated_test_data
            )

            logger.info("Loading predictor model...")
            predictor_model = load_predictor_model(predictor_dir_path)

            logger.info("Making predictions...")
            predictions_arr = predict_with_model(predictor_model, transformed_test_data)

            label_encoder = inference_pipeline.named_steps["target_encoder"]

            logger.info("Creating final predictions dataframe...")
            predictions_df = create_predictions_dataframe(
                pred_input=validated_test_data,
                predictions_arr=predictions_arr,
                prediction_field_name=prediction_field_name,
                label_encoder=label_encoder,
                data_schema=data_schema,
            )

            predictions_df = validate_predictions(
                predictions_df,
                data_schema,
            )

        logger.info("Saving predictions dataframe...")
        save_dataframe_as_csv(dataframe=predictions_df, file_path=predictions_file_path)

    except Exception as exc:
        err_msg = "Error occurred during prediction."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.PREDICT_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_batch_predictions()