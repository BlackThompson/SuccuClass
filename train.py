import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import time
import numpy as np
from torchinfo import summary
import os
import pandas as pd
import logging
from data.dataset import PlantDataset
from models.mobilenet_v2 import MobileNetV2
from models.cnn import SimpleCNN
from utils.trainer import train_model
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils.trainer import EarlyStopping
from utils.eval import evaluate_model
import datetime

# Set random seed
torch.manual_seed(42)


def main():
    # Define test configuration combinations
    model_configs = [
        # {"drop": 0},
        {"drop": 0.2},
        # {"wm": 0.75, "dm": 0.5},
        # {"wm": 0.75, "dm": 1.0},
        # {"wm": 1.0, "dm": 1.0},
        # {"wm": 1.5, "dm": 1.0},
        # {"wm": 1.5, "dm": 1.5},
        # {"wm": 1.5, "dm": 2.0},
    ]

    # Base parameter configuration
    base_params = {
        # "drop": 0.2,  # dropout
        "kz": 5,  # kernel_size
        "lr": 0.001,  # learning_rate
        "wm": 0.75,
        "dm": 0.75,
    }

    # Run each configuration in a loop
    for config in model_configs:
        # Merge base parameters with current configuration
        model_params = {**base_params, **config}

        # Configure logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_signature = f"wm{model_params['wm']}_dm{model_params['dm']}_drop{model_params['drop']}_kz{model_params['kz']}_lr{model_params['lr']}"
        log_filename = f"logs/training_{model_signature}_{timestamp}.log"

        # Create necessary directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        results_dir = f"results/{model_signature}_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)

        # Configure logging settings
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
            force=True,  # Ensure logging is reset for each loop
        )

        logging.info(f"Starting training with parameters: {model_params}")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Data preprocessing
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Create dataset
        dataset = PlantDataset(
            root_dir="data_augmentation",
            csv_file="classifications.csv",
            transform=transform,
        )

        # Split dataset
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        # Create data loader
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=2
        )

        # Create model
        num_classes = dataset.get_num_classes()
        model = MobileNetV2(
            num_classes=num_classes,
            width_mult=model_params["wm"],
            depth_mult=model_params["dm"],
            dropout=model_params["drop"],
            kernel_size=model_params["kz"],
        ).to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=model_params["lr"])

        # Train model
        checkpoint_path = f"checkpoints/model_{model_signature}_{timestamp}.pth"
        train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            checkpoint_path=checkpoint_path,
        )

        # Load best model and evaluate
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        metrics = evaluate_model(
            model, test_loader, device, model_signature, results_dir
        )

        # Save evaluation results
        eval_results_path = os.path.join(results_dir, f"evaluation_results.txt")
        with open(eval_results_path, "w") as f:
            f.write(f"Model Parameters:\n")
            for param, value in model_params.items():
                f.write(f"{param}: {value}\n")
            f.write("\nModel Statistics:\n")
            f.write(f"Model Size: {metrics['model_size_mb']:.2f} MB\n")
            f.write(f"Total Parameters: {metrics['total_parameters']:,}\n")
            f.write(f"Trainable Parameters: {metrics['trainable_parameters']:,}\n")
            f.write(
                f"Average Inference Time: {metrics['avg_inference_time']:.2f} ms per image\n"
            )
            f.write("\nEvaluation Results:\n")
            for metric, value in metrics.items():
                if metric not in [
                    "confusion_matrices",
                    "model_size_mb",
                    "total_parameters",
                    "trainable_parameters",
                ]:
                    f.write(f"{metric}: {value}\n")

        # Clean GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
