import torch
import torch.nn as nn
import math


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        if groups > 1:
            groups = in_planes
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, width_mult=1.0, kernel_size=3):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        def _make_divisible(v, divisor=8, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        inp = _make_divisible(inp)
        oup = _make_divisible(oup)
        hidden_dim = _make_divisible(round(inp * expand_ratio))

        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                # dw
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=hidden_dim,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        width_mult=1.0,
        depth_mult=1.0,
        dropout=0.2,
        kernel_size=3,
    ):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual

        # 确保所有的通道数都是 8 的倍数
        def _make_divisible(v, divisor=8, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # 确保向下舍入不会超过 10%
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        # 使用 _make_divisible 来调整所有通道数
        input_channel = _make_divisible(32 * width_mult)
        last_channel = _make_divisible(1280 * width_mult)

        # MobileNetV2的基础配置
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [ConvBNReLU(3, input_channel, kernel_size=kernel_size, stride=2)]

        # 构建中间的inverted residual blocks，确保所有通道数都对齐
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(
                c * width_mult
            )  # 使用 _make_divisible 确保对齐
            repeats = max(1, int(n * depth_mult))
            for i in range(repeats):
                stride = s if i == 0 else 1
                hidden_dim = _make_divisible(
                    input_channel * t
                )  # 确保扩展后的通道数也对齐
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        width_mult=width_mult,
                        kernel_size=kernel_size,
                    )
                )
                input_channel = output_channel

        # 构建最后几层
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(last_channel, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
