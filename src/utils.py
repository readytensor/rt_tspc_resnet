import json
import os
import random
import time
import threading
import psutil
import tracemalloc
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Union

import torch


def read_json_as_dict(input_path: str) -> Dict:
    """
    Reads a JSON file and returns its content as a dictionary.
    If input_path is a directory, the first JSON file in the directory is read.
    If input_path is a file, the file is read.

    Args:
        input_path (str): The path to the JSON file or directory containing a JSON file.

    Returns:
        dict: The content of the JSON file as a dictionary.

    Raises:
        ValueError: If the input_path is neither a file nor a directory,
                    or if input_path is a directory without any JSON files.
    """
    if os.path.isdir(input_path):
        # Get all the JSON files in the directory
        json_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".json")
        ]

        # If there are no JSON files, raise a ValueError
        if not json_files:
            raise ValueError("No JSON files found in the directory")

        # Else, get the path of the first JSON file
        json_file_path = json_files[0]

    elif os.path.isfile(input_path):
        json_file_path = input_path
    else:
        raise ValueError("Input path is neither a file nor a directory")

    # Read the JSON file and return it as a dictionary
    with open(json_file_path, "r", encoding="utf-8") as file:
        json_data_as_dict = json.load(file)

    return json_data_as_dict


def read_csv_in_directory(file_dir_path: str) -> pd.DataFrame:
    """
    Reads a CSV file in the given directory path as a pandas dataframe and returns
    the dataframe.

    Args:
    - file_dir_path (str): The path to the directory containing the CSV file.

    Returns:
    - pd.DataFrame: The pandas dataframe containing the data from the CSV file.

    Raises:
    - FileNotFoundError: If the directory does not exist.
    - ValueError: If no CSV file is found in the directory or if multiple CSV files are
        found in the directory.
    """
    if not os.path.exists(file_dir_path):
        raise FileNotFoundError(f"Directory does not exist: {file_dir_path}")

    csv_files = [file for file in os.listdir(
        file_dir_path) if file.endswith(".csv")]

    if not csv_files:
        raise ValueError(f"No CSV file found in directory {file_dir_path}")

    if len(csv_files) > 1:
        raise ValueError(
            f"Multiple CSV files found in directory {file_dir_path}.")

    csv_file_path = os.path.join(file_dir_path, csv_files[0])
    df = pd.read_csv(csv_file_path)
    return df


def read_tuning_datasets(tuning_dir_path: str) -> List[pd.DataFrame]:
    results = []
    datasets = [
        i
        for i in os.listdir(tuning_dir_path)
        if os.path.isdir(os.path.join(tuning_dir_path, i))
    ]

    datasets_paths = [os.path.join(tuning_dir_path, dataset)
                      for dataset in datasets]
    for path in datasets_paths:
        data = read_csv_in_directory(path)
        results.append(data)
    return results


def cast_time_col(data: pd.DataFrame, time_col: str, dtype: str) -> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): dataframe
        time_col (str): name of the time field
        dtype (str): type of time field ('INT', 'DATE', or 'DATETIME')

    Returns:
        pd.DataFrame: updated dataframe with time col appropriately cast
    """
    data = data.copy()
    if dtype == "INT":
        data[time_col] = data[time_col].astype(int)
    elif dtype in ["DATETIME", "DATE"]:
        data[time_col] = pd.to_datetime(data[time_col])
    else:
        raise ValueError(f"Invalid data type for time column: {dtype}")
    return data


def set_seeds(seed_value: int) -> None:
    """
    Set the random seeds for Python, NumPy, etc. to ensure
    reproducibility of results.

    Args:
        seed_value (int): The seed value to use for random
            number generation. Must be an integer.

    Returns:
        None
    """
    if isinstance(seed_value, int):
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
    else:
        raise ValueError(
            f"Invalid seed value: {seed_value}. Cannot set seeds.")


def train_test_split(
    data: pd.DataFrame,
    id_col: str,
    test_split: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a numpy array into train and test sets along the first dimension.

    Args:
    data (pd.DataFrame): The input dataframe.
    id_col (str): The name of the column containing the IDs.
    test_split (float): Fraction of the dataset to be included in the test split. Default is 0.2.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Two pandas dataframes representing the train and test sets.

    Raises:
    ValueError: If test_split is not between 0 and 1.

    """
    # Check if test_split value is valid
    if not 0 <= test_split <= 1:
        raise ValueError("test_split must be between 0 and 1")

    n_series = data[id_col].nunique()

    if n_series == 1:  # If only one series, split by index
        test_size = int(len(data) * test_split)
        test_set = data.iloc[-test_size:]
        train_set = data.iloc[:-test_size]

    else:  # If multiple series, split by series
        n_test_series = int(n_series * test_split)

        smallest_n_series = (
            data[id_col]
            .value_counts()
            .sort_values()
            .iloc[:n_test_series]
            .index.tolist()
        )

        test_set = data[data[id_col].isin(smallest_n_series)]
        train_set = data[~data[id_col].isin(smallest_n_series)]

    return train_set, test_set


def save_dataframe_as_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    """
    Saves a pandas dataframe to a CSV file in the given directory path.
    Float values are saved with 4 decimal places.

    Args:
    - df (pd.DataFrame): The pandas dataframe to be saved.
    - file_path (str): File path and name to save the CSV file.

    Returns:
    - None

    Raises:
    - IOError: If an error occurs while saving the CSV file.
    """
    try:
        dataframe.to_csv(file_path, index=False, float_format="%.4f")
    except IOError as exc:
        raise IOError(f"Error saving CSV file: {exc}") from exc


def save_json(file_path_and_name: str, data: Any) -> None:
    """Save json to a path (directory + filename)"""
    with open(file_path_and_name, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            default=lambda o: make_serializable(o),
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
        )


def make_serializable(obj: Any) -> Union[int, float, List[Union[int, float]], Any]:
    """
    Converts a given object into a serializable format.

    Args:
    - obj: Any Python object

    Returns:
    - If obj is an integer or numpy integer, returns the integer value as an int
    - If obj is a numpy floating-point number, returns the floating-point value
        as a float
    - If obj is a numpy array, returns the array as a list
    - Otherwise, uses the default behavior of the json.JSONEncoder to serialize obj

    """
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return json.JSONEncoder.default(None, obj)


def get_peak_memory_usage():
    """Returns the peak memory usage by current cuda device in (in MB) if available"""
    if not torch.cuda.is_available():
        return 0
    current_device = torch.cuda.current_device()
    peak_memory = torch.cuda.max_memory_allocated(current_device)
    return peak_memory / (1024 * 1024)


class ResourceTracker(object):
    """
    This class serves as a context manager to track time and
    memory allocated by code executed inside it.
    """

    def __init__(self, logger, monitoring_interval):
        self.logger = logger
        self.monitor = MemoryMonitor(
            logger=logger, interval=monitoring_interval)

    def __enter__(self):
        self.start_time = time.time()
        tracemalloc.start()
        self.monitor.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.monitor.stop()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed_time = self.end_time - self.start_time
        peak_python_memory_mb = peak / 1024**2
        process_cpu_peak_memory_mb = self.monitor.get_peak_memory_usage()
        gpu_peak_memory_mb = get_peak_memory_usage()

        self.logger.info(f"Execution time: {elapsed_time:.2f} seconds")
        self.logger.info(
            f"Peak Python Allocated Memory: {peak_python_memory_mb:.2f} MB"
        )
        self.logger.info(
            f"Peak CUDA GPU Memory Usage (Incremental): {gpu_peak_memory_mb:.2f} MB"
        )
        self.logger.info(
            f"Peak System RAM Usage (Incremental): {process_cpu_peak_memory_mb:.2f} MB"
        )


class MemoryMonitor:
    initial_cpu_memory = None
    peak_cpu_memory = 0  # Class variable to store peak memory usage

    def __init__(self, interval=20.0, logger=print):
        self.interval = interval
        self.logger = logger or print
        self.running = False
        self.thread = threading.Thread(target=self.monitor_loop)

    def monitor_memory(self):
        process = psutil.Process(os.getpid())
        total_memory = process.memory_info().rss

        # Check if the current memory usage is a new peak and update accordingly
        self.peak_cpu_memory = max(self.peak_cpu_memory, total_memory)
        if self.initial_cpu_memory is None:
            self.initial_cpu_memory = self.peak_cpu_memory

    def monitor_loop(self):
        """Runs the monitoring process in a loop."""
        while self.running:
            self.monitor_memory()
            time.sleep(self.interval)

    def _schedule_monitor(self):
        """Internal method to schedule the next execution"""
        self.monitor_memory()
        # Only reschedule if the timer has not been canceled
        if self.timer is not None:
            self.timer = threading.Timer(self.interval, self._schedule_monitor)
            self.timer.start()

    def start(self):
        """Starts the memory monitoring."""
        if not self.running:
            self.running = True
            self.thread.start()

    def stop(self):
        """Stops the periodic monitoring"""
        self.running = False
        self.thread.join()  # Wait for the monitoring thread to finish

    def get_peak_memory_usage(self):
        # Convert both CPU and GPU memory usage from bytes to megabytes
        incremental_cpu_peak_memory = (
            self.peak_cpu_memory - self.initial_cpu_memory
        ) / (1024**2)

        return incremental_cpu_peak_memory

    @classmethod
    def get_peak_memory(cls):
        """Returns the peak memory usage"""
        return cls.peak_cpu_memory
