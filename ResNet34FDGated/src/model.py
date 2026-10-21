import torch
import sys
import os
import torch.nn as nn
import torchvision.models as models
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY, DEVICE, MODEL_SAVE_DIR, RESULTS_DIR
from config import DEVICE

class ImprovedResNet34(nn.Module):
    
    def __init__(self, pretrained=True):
        super(ImprovedResNet34, self).__init__()
        # Load the pretrained ResNet-34 model
        # Use the new weights argument instead of the deprecated pretrained argument
        if pretrained:
            # Load ImageNet-pretrained weights
            self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet34(weights=None)
        
        # Remove the final fully connected layer and retain feature extraction
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Add spatial attention
        # Input: (batch_size, 512, 7, 7) - output of the final ResNet34 convolutional layer
        # Output: (batch_size, 512, 7, 7) - weighted feature maps
        self.attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),  # Reduce to a single-channel attention map
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
        # Create one improved ResNet-34 model to process one image
        self.image_model = ImprovedResNet34(pretrained=pretrained).to(DEVICE)
        
        # Image feature dimension (ResNet-34 outputs 512-dimensional features)
        self.image_feature_dim = 512
        # Numerical feature dimension
        self.data_feature_dim = 3  # 3 numerical features
        
        # Numerical feature processing module - expand 3-dimensional features to 512 dimensions
        self.data_processor = nn.Sequential(
            nn.Linear(self.data_feature_dim, 512),  # Use a variable instead of a hard-coded value
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Prevent overfitting
            nn.Linear(512, self.image_feature_dim),  # Second step: 512→512 dimensions
            nn.BatchNorm1d(self.image_feature_dim),
            nn.ReLU()
        )

        # Feature-wise Gated Fusion: generate adaptive weights for each numerical feature using both modalities.
        # The input concatenates F_img(512) and F_num(512), and the output has the same dimension as F_num.
        self.gate_network = nn.Sequential(
            nn.Linear(self.image_feature_dim * 2, self.image_feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.image_feature_dim // 2, self.image_feature_dim),
            nn.Sigmoid()
        )
        
        # Fully connected layers fuse all features (updated input dimension)
        self.fc_layers = nn.Sequential(
            # Modified: image features (512) + expanded numerical features (512) = 1024 dimensions
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

        # G = Sigmoid(MLP_gate(Concat(F_img, F_num))), so weights lie in [0, 1].
        gate_input = torch.cat([img_features, data_features], dim=1)
        gate_weights = self.gate_network(gate_input)
        gated_data_features = gate_weights * data_features
        
        # F_fused = Concat(F_img, G ⊙ F_num) remains 1024-dimensional.
        combined_features = torch.cat([img_features, gated_data_features], dim=1)
        
        # Produce outputs through the fully connected layers
        output = self.fc_layers(combined_features)
        
        return output  # Classification does not require squeeze
