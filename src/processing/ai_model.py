import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision.models import ResNet18_Weights
# 1. We define the architecture (It must be identical to the dummy we created)
from torchvision import models

class BreastCancerCNN(nn.Module):
    def __init__(self, num_classes=2, use_weights=False): # Set to False by default for inference
        super(BreastCancerCNN, self).__init__()
        weights = ResNet18_Weights.DEFAULT if use_weights else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, 
                                        padding=3, bias=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def predict_cancer(image_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BreastCancerCNN()
        model_path = os.path.join('models', 'breast_cancer_init.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        image = Image.open(image_path).convert('L')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            # ERROR SOLUTION: Convert to probability and select the 'malignant' class
            probabilities = torch.softmax(output, dim=1)
            prediction_score = probabilities[0][1].item()
        
        return round(prediction_score * 100, 2)

    except Exception as e:
        print(f"Error en IA: {e}")
        return None