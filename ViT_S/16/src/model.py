

import timm
import torch.nn as nn


MODEL_NAME = 'vit_small_patch16_224.augreg_in1k'


class ViTSmall16(nn.Module):
    

    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        if num_classes != 3:
            raise ValueError(f'ViT-S/16 is fixed to 3 classes in this experiment; received num_classes={num_classes}.')
        self.model = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            num_classes=3,
        )

    def forward(self, images):
        return self.model(images)
