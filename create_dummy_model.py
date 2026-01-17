import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
from pathlib import Path

class BreastCancerCNN(nn.Module):
    """
    CNN avanzada para detección de cáncer de mama.
    Basada en ResNet18 con transfer learning.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(BreastCancerCNN, self).__init__()
        
        # Usar ResNet18 preentrenado como base
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modificar primera capa para imágenes en escala de grises
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, 
                                        padding=3, bias=False)
        
        # Obtener número de features de la última capa
        num_features = self.backbone.fc.in_features
        
        # Reemplazar clasificador final con una arquitectura personalizada
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


class CustomCNN(nn.Module):
    """
    CNN personalizada desde cero para comparación.
    """
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        
        # Bloque convolucional 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Bloque convolucional 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Bloque convolucional 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Bloque convolucional 4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Capa adaptativa para manejar diferentes tamaños de entrada
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


class BreastCancerDataset(Dataset):
    """
    Dataset personalizado para imágenes de mamografías.
    Esperado: carpetas 'benign' y 'malignant' con imágenes.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = ['benign', 'malignant']
        
        # Cargar rutas de archivos y etiquetas
        for idx, class_name in enumerate(self.classes):
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.dcm']:
                        self.samples.append((str(img_path), idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Cargar imagen (asumiendo PNG/JPG, para DICOM necesitarías pydicom)
        image = Image.open(img_path).convert('L')  # Escala de grises
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=224, augment=True):
    """Retorna transformaciones para entrenamiento y validación."""
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, val_transform


def train_model(model, train_loader, val_loader, num_epochs=50, 
                learning_rate=0.001, device='cuda'):
    """Función de entrenamiento completa."""
    model = model.to(device)
    
    # Loss balanceado para clases desbalanceadas (común en cáncer)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer con weight decay para regularización
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # Scheduler step
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'models/best_breast_cancer_model.pth')
            print(f'Mejor modelo guardado con val_acc: {val_acc:.2f}%')


if __name__ == "__main__":
    # Configuración
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {DEVICE}")
    
    # Crear modelo (elige uno)
    # Opción 1: Transfer learning (recomendado)
    model = BreastCancerCNN(num_classes=2, pretrained=True)
    
    # Opción 2: CNN desde cero
    # model = CustomCNN(num_classes=2)
    
    # Guardar arquitectura inicial
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/breast_cancer_init.pth")
    print("Modelo inicializado y guardado en /models")
    
    print("\nArquitectura del modelo:")
    print(model)
    print(f"\nTotal parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Para entrenar (descomenta y ajusta las rutas):
    """
    train_transform, val_transform = get_transforms(img_size=224, augment=True)
    
    train_dataset = BreastCancerDataset('data/train', transform=train_transform)
    val_dataset = BreastCancerDataset('data/val', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    train_model(model, train_loader, val_loader, num_epochs=50, device=DEVICE)
    """