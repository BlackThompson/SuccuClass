import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_training_history(csv_file):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()

    # Save plot
    plot_path = os.path.join("metrics", "training_plot.png")
    plt.savefig(plot_path)
    plt.close()


# Usage:
# plot_training_history('metrics/training_history_YYYYMMDD_HHMMSS.csv')
