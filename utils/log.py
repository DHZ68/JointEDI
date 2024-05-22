import logging
import time
import os
import json

def setup_logging():
    """Set up logging"""
    current_time = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("log", current_time)  # Create log folder based on current time

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    metric_log_filename = os.path.join(log_dir, "metric.log")
    time_log_filename = os.path.join(log_dir, "time.log")

    # Create two loggers
    logger = logging.getLogger("metricLogger")
    logger.setLevel(logging.INFO)
    metric_handler = logging.FileHandler(metric_log_filename, mode="a")
    metric_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
        )
    )
    logger.addHandler(metric_handler)
    logger.addHandler(logging.StreamHandler())

    time_logger = logging.getLogger("timeLogger")
    time_logger.setLevel(logging.INFO)
    time_handler = logging.FileHandler(time_log_filename, mode="a")
    time_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
    )
    time_logger.addHandler(time_handler)

    return logger, time_logger, log_dir

def save_config(args, log_dir):
    """Save configuration parameters to JSON file"""
    config_path = os.path.join(log_dir, 'config.json')
    with open(config_path, 'w') as config_file:
        json.dump(args, config_file, indent=4)

logger, time_logger, log_dir = setup_logging()
