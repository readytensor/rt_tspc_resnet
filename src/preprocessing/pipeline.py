import joblib
import pandas as pd
from typing import Any, Tuple
from sklearn.pipeline import Pipeline
from preprocessing import custom_transformers as transformers


def create_preprocess_pipelines(
    data_schema: Any,
    preprocessing_config: dict,
    encode_len: int,
) -> Tuple[Pipeline, Pipeline]:
    """
    Constructs two preprocessing pipeline for time-series data:
        one for training and another inference.

    Args:
        data_schema: The schema of the data, containing information like column names.
        preprocessing_config (dict): Configuration parameters for preprocessing, like scaler bounds.
        encode_len (int): The length of the encoding window.

    Returns:
        Tuple[Pipeline, Pipeline]: Tuple of training and inference pipelines
    """
    scaler = transformers.TimeSeriesMinMaxScaler(columns=data_schema.features)

    # Common steps for both train and inference pipelines
    common_steps = [
        (
            "column_selector",
            transformers.ColumnSelector(
                columns=(
                    [data_schema.id_col, data_schema.time_col, data_schema.target]
                    + data_schema.features
                )
            ),
        ),
        (
            "time_col_caster",
            transformers.TimeColCaster(
                time_col=data_schema.time_col, data_type=data_schema.time_col_dtype
            ),
        ),
        (
            "df_sorter",
            transformers.DataFrameSorter(
                sort_columns=[data_schema.id_col, data_schema.time_col],
                ascending=[True, True],
            ),
        ),
        (
            "padding",
            transformers.PaddingTransformer(
                id_col=data_schema.id_col, target_col=data_schema.target
            ),
        ),
        ("minmax_scaler", scaler),
    ]
    training_steps = common_steps.copy()
    inference_steps = common_steps.copy()
    # training-specific steps

    training_steps.extend(
        [
            (
                "target_encoder",
                transformers.LabelEncoder(columns=[data_schema.target]),
            ),
            (
                "reshaped_3d",
                transformers.ReshaperToThreeD(
                    id_col=data_schema.id_col,
                    time_col=data_schema.time_col,
                    value_columns=data_schema.features,
                    target_column=data_schema.target,
                ),
            ),
            (
                "window_generator",
                transformers.TimeSeriesWindowGenerator(
                    window_size=encode_len,
                    stride=1,
                    max_windows=preprocessing_config["max_windows"],
                ),
            ),
        ]
    )
    # inference-specific steps
    inference_steps.extend(
        [
            (
                "reshaped_3d",
                transformers.ReshaperToThreeD(
                    id_col=data_schema.id_col,
                    time_col=data_schema.time_col,
                    value_columns=data_schema.features,
                    target_column=data_schema.target,
                ),
            ),
            (
                "window_generator",
                transformers.TimeSeriesWindowGenerator(
                    window_size=encode_len,
                    stride=encode_len // 2,
                    max_windows=None,
                    mode="inference",
                ),
            ),
        ]
    )
    return Pipeline(training_steps), Pipeline(inference_steps)


def train_pipeline(pipeline: Pipeline, data: pd.DataFrame) -> pd.DataFrame:
    """
    Train the preprocessing pipeline.

    Args:
        pipeline (Pipeline): The preprocessing pipeline.
        data (pd.DataFrame): The training data as a pandas DataFrame.

    Returns:
        Pipeline: Fitted preprocessing pipeline.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pd.DataFrame")
    pipeline.fit(data)
    return pipeline


def transform_inputs(pipeline: Pipeline, input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input data using the preprocessing pipeline.

    Args:
        pipeline (Pipeline): The preprocessing pipeline.
        input_data (pd.DataFrame): The input data as a pandas DataFrame.

    Returns:
        pd.DataFrame: The transformed data.
    """
    return pipeline.transform(input_data)


def save_pipeline(pipeline: Pipeline, file_path_and_name: str) -> None:
    """Save the fitted pipeline to a pickle file.

    Args:
        pipeline (Pipeline): The fitted pipeline to be saved.
        file_path_and_name (str): The path where the pipeline should be saved.
    """
    joblib.dump(pipeline, file_path_and_name)


def load_pipeline(file_path_and_name: str) -> Pipeline:
    """Load the fitted pipeline from the given path.

    Args:
        file_path_and_name: Path to the saved pipeline.

    Returns:
        Fitted pipeline.
    """
    return joblib.load(file_path_and_name)
