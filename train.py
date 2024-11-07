import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import time
import json
import numpy as np
from model import MobileNetV2
from config import ModelConfig
from utils import setup_logger, save_metrics
from tqdm import tqdm
import os


def get_optimizer(config, model_parameters):
    if config.optimizer_type.lower() == "adam":
        return optim.Adam(
            model_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_type.lower() == "sgd":
        return optim.SGD(
            model_parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")


def get_scheduler(config, optimizer):
    if not config.use_lr_scheduler:
        return None

    if config.lr_scheduler_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_scheduler_step_size,
            gamma=config.lr_scheduler_gamma,
        )
    elif config.lr_scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif config.lr_scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_scheduler_gamma,
            patience=config.patience // 2,
            verbose=True,
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {config.lr_scheduler_type}")


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return total_loss / len(data_loader), predictions, targets


def calculate_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def train_model(config, logger):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data loading and preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Load and split dataset
    # this dataset is already given with an 80-10-10 split
    logger.info("Loading Plantnet-300K dataset...")
    train_dataset = datasets.ImageFolder("./data/plantnet_300K/images/train", transform=transform)
    test_dataset = datasets.ImageFolder("./data/plantnet_300K/images/test", transform=transform)
    val_dataset = datasets.ImageFolder("./data/plantnet_300K/images/val", transform=transform)

    # only use 10% of the data
    train_dataset = torch.utils.data.Subset(
        train_dataset, range(len(train_dataset) // 10)
    )
    test_dataset = torch.utils.data.Subset(
        test_dataset, range(len(test_dataset) // 10)
    )
    val_dataset = torch.utils.data.Subset(
        val_dataset, range(len(val_dataset) // 10)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Initialize model, criterion, optimizer and scheduler
    model = MobileNetV2(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_scheduler(config, optimizer)

    # Initialize metrics tracking
    metrics = {"history": [], "final_metrics": {}}

    best_val_loss = float("inf")
    patience_counter = 0

    # Log model information
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Model size: {calculate_model_size(model):.2f} MB")

    # Training loop
    logger.info("Starting training...")
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Training phase
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100. * train_correct / train_total:.2f}%",
                }
            )

        # Validation phase
        val_loss, val_preds, val_targets = evaluate(
            model, val_loader, criterion, device
        )
        val_accuracy = sum(np.array(val_preds) == np.array(val_targets)) / len(
            val_targets
        )

        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Log metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        metrics["history"].append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        logger.info(f"Epoch {epoch+1}/{config.epochs}:")
        logger.info(
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}"
        )
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        logger.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                "best_model.pth",
            )
            logger.info(f"Saved new best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"Patience counter: {patience_counter}")
            if patience_counter >= config.patience:
                logger.info("Early stopping triggered")
                break

    # Load best model for final evaluation
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final evaluation on test set
    test_loss, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )
    test_accuracy = sum(np.array(test_preds) == np.array(test_targets)) / len(
        test_targets
    )

    # Calculate final metrics
    conf_mat = confusion_matrix(test_targets, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_targets, test_preds, average="weighted"
    )

    # Calculate inference time
    start_time = time.time()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            _ = model(images)
    inference_time = (time.time() - start_time) / len(test_loader.dataset)

    # Save final metrics
    metrics["final_metrics"] = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "inference_time_ms": float(inference_time * 1000),
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "model_size_mb": float(calculate_model_size(model)),
        "confusion_matrix": conf_mat.tolist(),
        "best_validation_loss": float(best_val_loss),
    }

    # Save metrics to files
    history_file, final_metrics_file = save_metrics(metrics)
    logger.info(f"Training history saved to: {history_file}")
    logger.info(f"Final metrics saved to: {final_metrics_file}")

    # Log final results
    logger.info("\nFinal Test Results:")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Inference Time per Sample: {inference_time*1000:.2f} ms")
    logger.info(f"Model Size: {calculate_model_size(model):.2f} MB")
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{conf_mat}")

    return test_accuracy


def main():
    # Load configuration
    config = ModelConfig()
    logger = setup_logger()

    # Save configuration
    config_dict = {k: v for k, v in config.__dict__.items()}

    # create metrics/config.json
    os.makedirs("metrics", exist_ok=True)

    with open("metrics/config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    # Train model
    final_accuracy = train_model(config, logger)
    logger.info(f"Training completed with final accuracy: {final_accuracy:.4f}")


if __name__ == "__main__":
    main()
