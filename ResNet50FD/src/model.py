import torch
import sys
import os
import torch.nn as nn
import torchvision.models as models
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY, DEVICE, MODEL_SAVE_DIR, RESULTS_DIR
from config import DEVICE

class ImprovedResNet50(nn.Module):
    
    def __init__(self, pretrained=True):
        super(ImprovedResNet50, self).__init__()
        # Load the pretrained ResNet-50 model
        # Use the new weights argument instead of the deprecated pretrained argument
        if pretrained:
            # Load ImageNet-pretrained weights
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.resnet = models.resnet50(weights=None)
        
        # Remove the final fully connected layer and retain feature extraction
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Add spatial attention
        # Input: (batch_size, 2048, 7, 7) - output of the final ResNet50 convolutional layer
        # Output: (batch_size, 2048, 7, 7) - weighted feature maps
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 1, kernel_size=1),  # Reduce to a single-channel attention map
            nn.Sigmoid()  # Normalize to the [0,1] range
        )
        
        # Freeze the initial convolutional parameters (transfer learning strategy)
        # Keep the last 10 layers trainable to balance feature reuse and task adaptation
        for param in list(self.features.parameters())[:-10]:
            param.requires_grad = False

    def forward(self, x):
        # x shape: [batch_size, 3, 224, 224]
        x = self.features(x)
        # Apply attention
        attn = self.attention(x)
        x = x * attn
        # Global average pooling
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # Flatten features
        x = x.view(x.size(0), -1)
        return x


class MultiModalModel(nn.Module):
    
    def __init__(self, num_classes, pretrained=True):
        super(MultiModalModel, self).__init__()
        # Create one improved ResNet-50 model to process one image
        self.image_model = ImprovedResNet50(pretrained=pretrained).to(DEVICE)
        
        # Image feature dimension (ResNet-50 outputs 2048-dimensional features)
        self.image_feature_dim = 2048
        # Numerical feature dimension
        self.data_feature_dim = 3  # 3 numerical features
        
        # Numerical feature processing module - expand 3-dimensional features to 2048 dimensions
        self.data_processor = nn.Sequential(
            nn.Linear(self.data_feature_dim, 512),  # Use a variable instead of a hard-coded value
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Prevent overfitting
            nn.Linear(512, self.image_feature_dim),  # Second step: 512→2048 dimensions
            nn.BatchNorm1d(self.image_feature_dim),
            nn.ReLU()
        )
        
        # Fully connected layers fuse all features (updated input dimension)
        self.fc_layers = nn.Sequential(
            # Image features (2048) + expanded numerical features (2048) = 4096 dimensions
            nn.Linear(self.image_feature_dim + self.image_feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevent overfitting
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, num_classes)  # Output layer for classification
        )

    def forward(self, images, data):
        # images shape: [batch_size, 3, 224, 224] (batch, channels, height, width)
        # Process one image
        img_features = self.image_model(images)
        
        # Added: process numerical features (3→512 dimensions)
        data_features = self.data_processor(data)
        
        # Concatenate image features and processed numerical features
        combined_features = torch.cat([img_features, data_features], dim=1)
        
        # Produce outputs through the fully connected layers
        output = self.fc_layers(combined_features)
        
        return output  # Classification does not require squeeze
