# MobileNetV2

A PyTorch implementation of MobileNetV2 for MNIST digit classification with comprehensive training metrics and visualization.

## Features

- Custom MobileNetV2 implementation from scratch
- Configurable model architecture and training parameters
- Comprehensive metrics tracking:
  - Confusion Matrix
  - Precision, Recall, F1 Score
  - Inference Time
  - Model Size
  - Parameter Count
- Early stopping
- Learning rate scheduling
- Training progress visualization
- Detailed logging system

## Requirements

```bash
torch
torchvision
numpy
scikit-learn
matplotlib
tqdm
```

## Project Structure

```
mobilenet_mnist/
├── data/           # MNIST dataset (downloaded automatically)
├── logs/           # Training logs
├── metrics/        # Training metrics and plots
├── models/         # Saved model checkpoints
├── model.py        # MobileNetV2 implementation
├── config.py       # Configuration parameters
├── train.py        # Training script
├── utils.py        # Utility functions
├── plot_results.py # Visualization tools
└── requirements.txt
```

## Configuration

The model and training parameters can be configured in `config.py`. Key parameters include:

```python
# Model Architecture Parameters
self.initial_channels = 32
self.num_classes = 10
self.dropout_rate = 0.2
self.width_multiplier = 1.0  # Controls the width of the network
self.expansion_factor = 6    # For inverted residuals

# Layer Configuration
self.inverted_residual_settings = [
    # t, c, n, s
    # t: expansion factor, c: output channels, n: repeat times, s: stride
    [1, 16, 1, 1],
    [6, 24, 2, 1],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]

# Training Parameters
self.learning_rate = 0.001
self.batch_size = 64
self.epochs = 50
self.patience = 5
```

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run training:

```bash
python train.py
```

3. Plot training results:

```bash
python plot_results.py
```

## Model Architecture

The MobileNetV2 implementation consists of:

- Inverted residual blocks
- Configurable width multiplier
- Dropout for regularization
- Batch normalization
- Adaptive pooling

See `model.py` for detailed implementation.

## Training Process

The training script includes:

- Train/validation/test split (8:1:1)
- Early stopping with configurable patience
- Learning rate scheduling
- Progress bars with tqdm
- Comprehensive metric logging
- Model checkpointing

## Metrics and Logging

Training progress and metrics are saved in:

- `logs/`: Detailed training logs
- `metrics/`: CSV files with training history
- `metrics/training_plot.png`: Loss visualization
- `metrics/final_metrics.json`: Final evaluation metrics

## Performance Monitoring

The training process logs:

- Per-batch loss and accuracy
- Validation metrics
- Learning rate changes
- Model size and parameter count
- Inference time measurements
