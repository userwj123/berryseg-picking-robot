from pathlib import Path

import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


def resolve_device(value):
    from blueberry_seg.runtime import device
    selected = device(value)
    if ',' in selected:
        raise ValueError('Branch prototype currently supports one GPU per process')
    return 'cpu' if selected=='cpu' else f'cuda:{selected}'


def build_model(pretrained=None):
    model = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None,
                                       num_classes=21, aux_loss=True)
    if pretrained is not None:
        state = torch.load(Path(pretrained), map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    model.classifier[-1] = nn.Conv2d(256,2,kernel_size=1)
    model.aux_classifier = None
    return model


def load_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != 'deeplabv3_mobilenet_v3_large_branch':
        raise ValueError('Not a supported branch segmentation checkpoint')
    model = build_model()
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), checkpoint


def train_mode(model, freeze_backbone):
    model.train()
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(not freeze_backbone)
    # Twenty images cannot provide reliable new batch-normalization statistics.
    # Frozen pretrained statistics also support small CPU batches safely.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
