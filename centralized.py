import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from opacus import PrivacyEngine

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        # FashionMNIST : 1 canal en niveaux de gris
        self.conv1 = nn.Conv2d(1, 6, 5)
        # GroupNorm requis pour la compatibilité avec Opacus (DP)
        self.gn1 = nn.GroupNorm(num_groups=2, num_channels=6)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=16)
        
        # FashionMNIST 28x28 -> après 2x(conv5+pool2) -> 4x4
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = torch.flatten(x, 1) # Flattens all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Downloads, transforms (to tensors and normalizes), and loads the FashionMNIST dataset
def load_data(node_id=0, num_clients=10, batch_size=64):
    """Load FashionMNIST partitionné par client (simulation non-IID réaliste)."""
    # 1 seul canal pour FashionMNIST (niveaux de gris)
    trf = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    trainset = FashionMNIST("./data", train=True, download=True, transform=trf)
    testset  = FashionMNIST("./data", train=False, download=True, transform=trf)

    # Partition IID simple : chaque client reçoit 1/num_clients du dataset
    # 60000 // 10 = 6000 images par client
    n = len(trainset) // num_clients
    indices = list(range(node_id * n, (node_id + 1) * n))
    client_trainset = torch.utils.data.Subset(trainset, indices)

    trainloader = DataLoader(client_trainset, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

# Initializes and returns an instance of the PyTorch model (moved to CPU or GPU)
def load_model():
    """Returns an instance of our Net model initialized and ready to run."""
    return Net().to(DEVICE)

# Calcule la taille d'un modèle en Mo
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

# Trains the global model with Local Differential Privacy and FedProx penalty
def train_fedprox_dp(net, global_params_dict, trainloader, epochs, mu=0.1):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    privacy_engine = PrivacyEngine()
    # On rend l'entraînement privé
    net_dp, optimizer_dp, trainloader_dp = privacy_engine.make_private(
        module=net,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
    )
    
    # Placer les poids globaux sur le bon appareil pour accélérer le calcul L2
    global_params_on_device = {k: v.to(DEVICE) for k, v in global_params_dict.items()}
    
    net_dp.train()
    for epoch in range(epochs):
        for images, labels in trainloader_dp:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer_dp.zero_grad()
            
            # Loss standard
            loss = criterion(net_dp(images), labels)
            
            # Pénalité FedProx : mu/2 * ||w - w_t||^2
            proximal_term = 0.0
            for param_name, param in net_dp.named_parameters():
                original_name = param_name.replace("_module.", "")
                # FedGN : on ne pénalise pas les couches GroupNorm car elles sont strictement locales
                if "gn" not in original_name and original_name in global_params_on_device:
                    proximal_term += torch.sum((param - global_params_on_device[original_name]) ** 2)
            
            total_loss = loss + (mu / 2.0) * proximal_term
            total_loss.backward()
            optimizer_dp.step()
            
    # On recharge les poids dans le réseau original pour retirer le wrapper de PrivacyEngine
    net.load_state_dict(net_dp._module.state_dict())
    
    # Extraction de l'Epsilon consommé (avec un delta standard de 1e-5)
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    
    return net, epsilon

# Evaluates the model on the test set without tracking gradients (no_grad) for performance
def test(net, testloader, device=None):
    """Validate the model on the test set."""
    if device is None:
        device = DEVICE
        
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.max(1)[1] == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

if __name__ == "__main__":
    print(f"Hardware computing initialized on: {DEVICE}")
    model = load_model()
    trainloader, testloader = load_data()
    
    print("Starting FedProx+DP training...")
    # Simulation du dictionnaire du serveur (poids initiaux)
    global_params = {k: v.clone() for k, v in model.state_dict().items()}
    model, _ = train_fedprox_dp(model, global_params, trainloader, epochs=2)
    
    print("Starting evaluation...")
    loss, accuracy = test(model, testloader)
    print(f"Final test loss: {loss:.4f}")
    print(f"Final test accuracy: {accuracy:.4f}")
