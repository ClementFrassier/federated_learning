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
        # FashionMNIST a 1 seul canal de couleur (niveaux de gris)
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Ajout de GroupNorm pour FedGN (compatible avec le DP d'Opacus)
        self.gn1 = nn.GroupNorm(num_groups=2, num_channels=6)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=16)
        
        # FashionMNIST fait 28x28, donc après 2 convs et 2 pools, on a 4x4
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
def load_data(node_id=0, num_clients=10, batch_size=128):
    """Load FashionMNIST partitionné par client."""
    # 1 seul canal pour la normalisation de FashionMNIST
    trf = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    trainset = FashionMNIST("./data", train=True, download=True, transform=trf)
    testset  = FashionMNIST("./data", train=False, download=True, transform=trf)

    # Partition IID simple : chaque client reçoit 1/num_clients du dataset
    n = len(trainset) // num_clients          # 60000 // 10 = 6000 images par client
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

# Sparsifie les paramètres (Magnitude Pruning)
def apply_sparsification(parameters, sparsity_ratio=0.5):
    """
    Apply magnitude pruning to keep only the top (1 - sparsity_ratio) weights.
    Returns the sparsified weights.
    """
    sparse_params = []
    for param in parameters:
        if param.size > 0:
            threshold = np.percentile(np.abs(param), sparsity_ratio * 100)
            param_copy = param.copy()
            param_copy[np.abs(param_copy) < threshold] = 0.0
            sparse_params.append(param_copy)
        else:
            sparse_params.append(param)
    return sparse_params

# Trains the models with Ditto approach (Global DP + Local Proximal)
def train_ditto_dp(global_net_dp, global_optimizer_dp, trainloader_dp, local_net, epochs, mu=0.01):
    """
    Ditto trains two models:
    - global_net: Trained with standard SGD (Opacus DP) to be sent to server.
    - local_net: Trained with standard SGD + Proximal penalty towards global_net.
    """
    criterion = torch.nn.CrossEntropyLoss()
    local_optimizer = torch.optim.SGD(local_net.parameters(), lr=0.001, momentum=0.9)

    # Save the initial global weights for the proximal penalty (Ditto local objective)
    # They should be on DEVICE
    global_params_on_device = {
        k: v.clone().to(DEVICE) 
        for k, v in global_net_dp._module.state_dict().items()
    }
    
    global_net_dp.train()
    local_net.train()
    
    for epoch in range(epochs):
        for images, labels in trainloader_dp:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # --- 1. Train Global Model (DP-SGD, no proximal term) ---
            global_optimizer_dp.zero_grad()
            global_loss = criterion(global_net_dp(images), labels)
            global_loss.backward()
            global_optimizer_dp.step()
            
            # --- 2. Train Local Model (SGD + Proximal term towards initial global weights) ---
            local_optimizer.zero_grad()
            local_loss = criterion(local_net(images), labels)
            
            # Pénalité Proximal : mu/2 * ||v - w_t||^2
            proximal_term = sum(
                torch.sum((p - global_params_on_device[n]) ** 2)
                for n, p in local_net.named_parameters()
                if n in global_params_on_device
            )
            
            total_local_loss = local_loss + (mu / 2.0) * proximal_term
            total_local_loss.backward()
            local_optimizer.step()
            
    # Resynchroniser global_net (sans wrapper) depuis global_net_dp
    # (nécessaire pour set_parameters côté Flower)
    return global_net_dp._module, local_net

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
    global_model = load_model()
    local_model = load_model()
    local_model.load_state_dict(global_model.state_dict())
    
    trainloader, testloader = load_data()
    
    from opacus import PrivacyEngine
    privacy_engine = PrivacyEngine()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=0.001, momentum=0.9)
    global_model_dp, global_optimizer_dp, trainloader_dp = privacy_engine.make_private(
        module=global_model,
        optimizer=global_optimizer,
        data_loader=trainloader,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
    )
    
    print("Starting Ditto + DP training...")
    global_model, local_model = train_ditto_dp(
        global_model_dp, global_optimizer_dp, trainloader_dp,
        local_model, epochs=2, mu=0.01
    )
    
    print("Starting evaluation of local model...")
    loss, accuracy = test(local_model, testloader)
    print(f"Final test loss: {loss:.4f}")
    print(f"Final test accuracy: {accuracy:.4f}")
