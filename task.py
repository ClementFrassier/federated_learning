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

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    """CNN for FashionMNIST (1-channel, 28×28).
    GroupNorm layers replace BatchNorm for:
    - Opacus (DP-SGD) compatibility (no per-sample statistics).
    - FedBN-style local normalisation (GN weights are never sent to the server).
    """

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        # GroupNorm: local to each client — never aggregated by the server
        self.gn1   = nn.GroupNorm(num_groups=2, num_channels=6)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.gn2   = nn.GroupNorm(num_groups=4, num_channels=16)
        # FashionMNIST 28×28 → after 2×(conv5 + pool2) → 4×4
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_data(node_id: int = 0, num_clients: int = 10, batch_size: int = 64):
    """Load FashionMNIST and return the IID partition for a given client."""
    trf = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    trainset = FashionMNIST("./data", train=True,  download=True, transform=trf)
    testset  = FashionMNIST("./data", train=False, download=True, transform=trf)

    n       = len(trainset) // num_clients
    indices = list(range(node_id * n, (node_id + 1) * n))
    client_trainset = torch.utils.data.Subset(trainset, indices)

    trainloader = DataLoader(client_trainset, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(testset,         batch_size=batch_size, shuffle=False)
    return trainloader, testloader


def load_model() -> Net:
    """Instantiate a fresh Net and move it to the available device."""
    return Net().to(DEVICE)


def get_model_size(model) -> float:
    """Return model size in megabytes (parameters + buffers)."""
    param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 ** 2


# ── FedProx + Standard-DP Training ───────────────────────────────────────────
def train_fedprox_dp(
    net_dp,
    optimizer_dp,
    trainloader_dp,
    global_params_on_device: dict,
    epochs: int,
    mu: float = 0.1,
):
    """One federation round of FedProx + DP-SGD training.

    IMPORTANT — The PrivacyEngine is NOT created here.
    It must be created once in the client (train function) and kept alive
    across ALL rounds so that epsilon accumulates correctly.

    Args:
        net_dp:                 Opacus-wrapped model (persistent across rounds).
        optimizer_dp:           Opacus-wrapped SGD optimiser (persistent).
        trainloader_dp:         Opacus-wrapped DataLoader (persistent).
        global_params_on_device: Server weights (non-GN layers) for FedProx penalty.
        epochs:                 Number of local training epochs.
        mu:                     FedProx proximal penalty weight µ.

    Returns:
        net_dp._module  (unwrapped weights, ready to send back)
    """
    criterion = torch.nn.CrossEntropyLoss()

    net_dp.train()
    for _ in range(epochs):
        for images, labels in trainloader_dp:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer_dp.zero_grad()

            # Standard cross-entropy loss
            loss = criterion(net_dp(images), labels)

            # FedProx proximal term: µ/2 · ||w − w_t||²
            # GN layers are excluded (FedBN principle: normalisation is local)
            proximal_term = sum(
                torch.sum((param - global_params_on_device[name.replace("_module.", "")]) ** 2)
                for name, param in net_dp.named_parameters()
                if "gn" not in name and name.replace("_module.", "") in global_params_on_device
            )
            (loss + (mu / 2.0) * proximal_term).backward()
            optimizer_dp.step()

    return net_dp._module


# ── Evaluation ────────────────────────────────────────────────────────────────
def test(net, testloader, device=None):
    """Evaluate the model on the test set and return (loss, accuracy)."""
    if device is None:
        device = DEVICE
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss   += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.max(1)[1] == labels).sum().item()
    loss    /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
