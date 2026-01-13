import torch
import torch.nn as nn

class BreastCancerModel(nn.Module):
    """
    CNN Architecture for Breast Cancer Detection.
    Designed according to Module 5 specifications.
    """
    def __init__(self):
        super(BreastCancerModel, self).__init__()
        
        # Capa 1: Detecta bordes y formas básicas
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # Capa 2: Detecta texturas sospechosas
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Capas densas para la clasificación final
        # Tras dos poolings de 2x2, una imagen de 256x256 pasa a ser de 64x64
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 1) # Salida: 0 (Sano) o 1 (Anomalía)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64) # "Aplanamos" la imagen
        x = torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))
        return x

if __name__ == "__main__":
    model = BreastCancerModel()
    print("--- AI Model Architecture ---")
    print(model)