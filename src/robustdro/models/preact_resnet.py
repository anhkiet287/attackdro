"""PreAct ResNet-18 for CIFAR — the standard backbone in the robustness lit.

This is the widely-used pre-activation ResNet (He et al. 2016) as it appears in
Madry-style adversarial training and the locuslab/robust_union codebase. Keeping
the architecture identical to those references matters for fair baseline
reproduction (golden rule #3).

The model optionally standardizes inputs with CIFAR-10 mean/std INSIDE forward()
so that attacks and epsilons live in raw [0,1] pixel space. Default is no
normalization (matches robust_union).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


class PreActBlock(nn.Module):
    """Pre-activation basic block (expansion = 1)."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(
                in_planes, self.expansion * planes, 1, stride=stride, bias=False
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes: int = 10,
        normalize: bool = False,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    ):
        super().__init__()
        self.in_planes = 64
        self.normalize = normalize
        if normalize:
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def PreActResNet18(num_classes: int = 10, normalize: bool = False) -> PreActResNet:
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, normalize=normalize)


_ARCHES = {"preact_resnet18": PreActResNet18}


def build_model(cfg: dict) -> nn.Module:
    """Build a model from a config dict (cfg['model'] + cfg['dataset'])."""
    mcfg = cfg.get("model", {})
    arch = mcfg.get("arch", "preact_resnet18")
    if arch not in _ARCHES:
        raise ValueError(f"Unknown arch '{arch}'. Known: {list(_ARCHES)}")
    num_classes = cfg.get("dataset", {}).get("num_classes", 10)
    return _ARCHES[arch](num_classes=num_classes, normalize=mcfg.get("normalize", False))
