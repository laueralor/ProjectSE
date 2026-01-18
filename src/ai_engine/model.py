import torch
import torch.nn as nn

class BreastCancerModel(nn.Module):
    """
    CNN Architecture for Breast Cancer Detection.
    Designed according to Module 5 specifications.
    """
    def __init__(self):
        super(BreastCancerModel, self).__init__()
        
        # Layer 1: Detects edges and basic shapes
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # Layer 2: Detects suspicious textures
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dense layers for final classification
        # After two 2x2 poolings, a 256x256 image becomes 64x64
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 1) # Output: 0 (Healthy) or 1 (Anomaly)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64) # "Flatten" the image
        x = torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))
        return x

if __name__ == "__main__":
    model = BreastCancerModel()
    print("--- AI Model Architecture ---")
    print(model)