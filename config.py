class ModelConfig:
    def __init__(self):
        # Model Architecture Parameters
        self.initial_channels = 32
        self.num_classes = 10
        self.dropout_rate = 0.2
        self.width_multiplier = 1.0  # Controls the width of the network
        self.expansion_factor = 6  # For inverted residuals

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
        self.patience = 3  # just test, usually 5 or 10
        self.weight_decay = 0.0001

        # Data Parameters
        self.image_size = 32
        self.train_split = 0.8
        self.val_split = 0.1
        self.test_split = 0.1

        # Optimizer Parameters
        self.optimizer_type = "adam"  # 'adam' or 'sgd'
        self.momentum = 0.9  # for SGD
        self.beta1 = 0.9  # for Adam
        self.beta2 = 0.999  # for Adam

        # Learning Rate Scheduler
        self.use_lr_scheduler = True
        self.lr_scheduler_type = "cosine"  # 'step', 'cosine', 'plateau'
        self.lr_scheduler_step_size = 10
        self.lr_scheduler_gamma = 0.1
