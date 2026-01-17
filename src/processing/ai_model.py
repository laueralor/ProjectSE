import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# 1. Definimos la arquitectura (Debe ser idéntica a la del dummy que creamos)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=3)
        self.fc = nn.Linear(10 * 254 * 254, 1)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x

def predict_cancer(image_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Cargar el modelo
        model = SimpleCNN()
        model_path = os.path.join('models', 'model.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Modo evaluación

        # 3. Transformar imagen
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

        image = Image.open(image_path).convert('L')
        image = transform(image).unsqueeze(0).to(device)

        # 4. Predicción real
        with torch.no_grad():
            output = model(image)
            prediction_score = output.item()
        
        return round(prediction_score * 100, 2)

    except Exception as e:
        print(f"Error en IA: {e}")
        return None