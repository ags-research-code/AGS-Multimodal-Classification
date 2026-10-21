
import torch.nn as nn


class MultiModalModel(nn.Module):
    
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.data_feature_dim = 3
        self.data_hidden_dim = 512
        self.data_processor = nn.Sequential(
            nn.Linear(self.data_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.data_hidden_dim),
            nn.BatchNorm1d(self.data_hidden_dim),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(self.data_hidden_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, data):
        
        return self.fc_layers(self.data_processor(data))
