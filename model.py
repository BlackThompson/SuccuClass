import torch
import torch.nn as nn
import math


class InvertedResidual(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, expand_ratio, dropout_rate=0.0
    ):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Expansion
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )

        layers.extend(
            [
                # Depthwise
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Dropout
                nn.Dropout2d(dropout_rate),
                # Pointwise
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, config):
        super(MobileNetV2, self).__init__()
        input_channel = int(config.initial_channels * config.width_multiplier)
        last_channel = 1280

        # building first layer
        self.features = [
            nn.Sequential(
                nn.Conv2d(config.input_channels, input_channel, 3, 2, 1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6(inplace=True),
                nn.Dropout2d(config.dropout_rate),
            )
        ]

        # building inverted residual blocks
        for t, c, n, s in config.inverted_residual_settings:
            output_channel = int(c * config.width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        dropout_rate=config.dropout_rate,
                    )
                )
                input_channel = output_channel

        # building last several layers
        self.features.append(
            nn.Sequential(
                nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(last_channel),
                nn.ReLU6(inplace=True),
                nn.Dropout2d(config.dropout_rate),
            )
        )

        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(last_channel, config.num_classes),
        )

        # weight initialization
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
