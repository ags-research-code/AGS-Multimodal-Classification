

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


MODEL_NAME = 'efficientnet_b0'


class EfficientNetB0Classifier(nn.Module):
    

    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        if num_classes != 3:
            raise ValueError(
                f'EfficientNet-B0 is fixed to 3 classes in this experiment; received num_classes={num_classes}.'
            )

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, 3)

    def forward(self, images):
        return self.model(images)
