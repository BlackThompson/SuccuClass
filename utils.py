import logging
import os
import csv
from datetime import datetime
import json


def setup_logger(log_dir="logs"):
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # This will print to console as well
        ],
    )

    return logging.getLogger(__name__)


def save_metrics(metrics, metrics_dir="metrics"):
    # Create metrics directory if it doesn't exist
    os.makedirs(metrics_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save training history to CSV
    history_file = os.path.join(metrics_dir, f"training_history_{timestamp}.csv")
    with open(history_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch_data in metrics["history"]:
            writer.writerow(
                [epoch_data["epoch"], epoch_data["train_loss"], epoch_data["val_loss"]]
            )

    # Save final metrics to JSON
    final_metrics_file = os.path.join(metrics_dir, f"final_metrics_{timestamp}.json")
    with open(final_metrics_file, "w") as f:
        json.dump(metrics["final_metrics"], f, indent=4)

    return history_file, final_metrics_file
