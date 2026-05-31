from typing import Tuple

import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    device: torch.device | None = None,
) -> Tuple[nn.Module, int, int]:
    """Construye un modelo ResNet18 adaptado a `num_classes`.

    Devuelve: (model, trainable_params, frozen_params)
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Congelar backbone si se requiere
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes),
    )

    # Contar parámetros
    trainable_params = 0
    frozen_params = 0
    for param in model.parameters():
        n = param.numel()
        if param.requires_grad:
            trainable_params += n
        else:
            frozen_params += n

    if device is not None:
        model = model.to(device)

    return model, trainable_params, frozen_params
