import torch
import torch.nn as nn
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3)
        self.fc = nn.Linear(1, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(torch.flatten(self.conv(x), 1)))

if __name__ == "__main__":
    model = SimpleCNN()
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")
    print("Modelo dummy recreado en /models")