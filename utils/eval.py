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

# 设置随机种子
torch.manual_seed(42)


def evaluate_model(model, test_loader, device, model_signature, results_dir):
    """
    Evaluates the model performance on test data
    Returns metrics including accuracies, confusion matrices, and model statistics
    """
    model.eval()
    predictions = []
    true_labels = []
    inference_times = []

    # 计算模型大小和参数数量
    model_size = 0
    for param in model.parameters():
        model_size += param.nelement() * param.element_size()  # 字节为单位
    model_size = model_size / (1024 * 1024)  # 转换为MB

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Load and prepare classification data
    df = pd.read_csv("classifications.csv")

    # Get unique taxonomic levels
    unique_families = sorted(df.family.unique())
    unique_genera = sorted(df.genus.unique())
    unique_species = sorted(df.species.unique())

    # Create mapping dictionaries
    species_to_genus = dict(zip(df.species, df.genus))
    species_to_family = dict(zip(df.species, df.family))

    # Debug print
    print(f"Number of unique species: {len(unique_species)}")
    print(f"Number of unique genera: {len(unique_genera)}")
    print(f"Number of unique families: {len(unique_families)}")

    # Perform inference
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # 对批次中的每张图片单独计算推理时间
            for i in range(batch_size):
                single_image = images[i : i + 1]

                start_time = time.time()
                _ = model(single_image)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

            # 使用整个批次进行预测（为了效率）
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.clamp(0, len(unique_species) - 1)

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    def plot_confusion_matrix(y_true, y_pred, labels, level):
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # 转换为百分比
        cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        # 创建图形
        plt.figure(figsize=(15, 15))

        # 绘制热力图
        sns.heatmap(
            cm_percentage,
            annot=True,
            fmt=".1f",
            xticklabels=labels,
            yticklabels=labels,
            cmap="YlOrRd",
            vmin=0,
            vmax=100,
        )

        plt.title(f"Confusion Matrix ({level}) - Percentage")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=45)

        cbar = plt.gca().collections[0].colorbar
        cbar.set_label("Percentage (%)")

        plt.tight_layout()
        save_path = os.path.join(results_dir, f"confusion_matrix_{level.lower()}.png")
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        # 打印每个类别的样本数量
        print(f"\nSample counts for each {level}:")
        counts = pd.Series(y_true).value_counts()
        for label in labels:
            count = counts.get(label, 0)
            print(f"{label}: {count} samples")

    # 转换预测和真实标签到实际的分类名称
    pred_species = [unique_species[p] for p in predictions]
    true_species = [unique_species[t] for t in true_labels]

    # 获取对应的genus和family
    pred_genera = [species_to_genus[s] for s in pred_species]
    true_genera = [species_to_genus[s] for s in true_species]
    pred_families = [species_to_family[s] for s in pred_species]
    true_families = [species_to_family[s] for s in true_species]

    # 绘制三个层级的混淆矩阵
    plot_confusion_matrix(true_families, pred_families, unique_families, "Family")
    plot_confusion_matrix(true_genera, pred_genera, unique_genera, "Genus")
    plot_confusion_matrix(true_species, pred_species, unique_species, "Species")

    # 计算各个层级的准确率
    def calculate_accuracy(true, pred):
        return 100 * sum(t == p for t, p in zip(true, pred)) / len(true)

    family_acc = calculate_accuracy(true_families, pred_families)
    genus_acc = calculate_accuracy(true_genera, pred_genera)
    species_acc = calculate_accuracy(true_species, pred_species)
    avg_acc = species_acc
    avg_inference_time = np.mean(inference_times) * 1000

    print("\n" + "=" * 50)
    print("Hierarchical Classification Results:")
    print("=" * 50)
    print(f"\nFamily Accuracy: {family_acc:.2f}%")
    print(f"Number of Families: {len(unique_families)}")
    print(f"\nGenus Accuracy: {genus_acc:.2f}%")
    print(f"Number of Genera: {len(unique_genera)}")
    print(f"\nSpecies Accuracy: {species_acc:.2f}%")
    print(f"Number of Species: {len(unique_species)}")
    print(f"\nAverage Accuracy: {avg_acc:.2f}%")
    print(f"\nAverage Inference Time: {avg_inference_time:.2f} ms per batch")
    print("=" * 50)

    print("\n" + "=" * 50)
    print("Model Statistics:")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Average Inference Time: {avg_inference_time:.2f} ms per image")
    print("=" * 50)

    return {
        "family_accuracy": family_acc,
        "genus_accuracy": genus_acc,
        "species_accuracy": species_acc,
        "average_accuracy": avg_acc,
        "avg_inference_time": avg_inference_time,
        "model_size_mb": model_size,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "confusion_matrices": {
            "family": confusion_matrix(
                true_families, pred_families, labels=unique_families
            ),
            "genus": confusion_matrix(true_genera, pred_genera, labels=unique_genera),
            "species": confusion_matrix(
                true_species, pred_species, labels=unique_species
            ),
        },
    }
