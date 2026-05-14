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

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    """CNN for FashionMNIST (1-channel, 28x28).
    GroupNorm is required for Opacus (DP) compatibility.
    """
    def __init__(self) -> None:
        super(Net, self).__init__()
        # 1 input channel (grayscale)
        self.conv1 = nn.Conv2d(1, 6, 5)
        # GroupNorm required for Opacus compatibility (replaces BatchNorm)
        self.gn1 = nn.GroupNorm(num_groups=2, num_channels=6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=16)
        # FashionMNIST 28x28 -> after 2x(conv5+pool2) -> 4x4
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


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_data(node_id: int = 0, num_clients: int = 10, batch_size: int = 32):
    """Load FashionMNIST and return the client's IID partition.
    
    Each client receives 1/num_clients of the full training set (6000 images
    for the default of 10 clients).
    """
    trf = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    trainset = FashionMNIST("./data", train=True, download=True, transform=trf)
    testset  = FashionMNIST("./data", train=False, download=True, transform=trf)

    # Simple IID split: each client gets a contiguous slice
    n = len(trainset) // num_clients
    indices = list(range(node_id * n, (node_id + 1) * n))
    client_trainset = torch.utils.data.Subset(trainset, indices)

    trainloader = DataLoader(client_trainset, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader


def load_model() -> Net:
    """Instantiate a fresh Net and move it to the available device."""
    return Net().to(DEVICE)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_model_size(model) -> float:
    """Return model size in megabytes (parameters + buffers)."""
    param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 ** 2


def apply_sparsification(parameters, sparsity_ratio: float = 0.5):
    """Apply magnitude pruning to a list of NumPy weight arrays.
    
    Zeroes out weights whose absolute value falls below the
    (sparsity_ratio * 100)-th percentile, keeping only the top
    (1 - sparsity_ratio) fraction of weights.
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


# ── Ditto + Local DP Training ─────────────────────────────────────────────────
def train_ditto_dp(
    global_net,
    local_net,
    trainloader,
    epochs: int,
    mu: float = 0.05,
    lr: float = 0.001,
    momentum: float = 0.9,
    noise_multiplier: float = 1.5,
    max_grad_norm: float = 1.0,
    dp_delta: float = 1e-5,
):
    """Train both Ditto models for one federation round.

    - global_net: updated with DP-SGD (Opacus) — sent back to the server.
    - local_net:  updated with SGD + proximal penalty towards global weights
                  — kept locally for personalised inference.

    Returns:
        global_net, local_net, dp_epsilon
    """
    criterion = torch.nn.CrossEntropyLoss()

    global_optimizer = torch.optim.SGD(global_net.parameters(), lr=lr, momentum=momentum)
    local_optimizer  = torch.optim.SGD(local_net.parameters(),  lr=lr, momentum=momentum)

    privacy_engine = PrivacyEngine()
    # Wrap the global model with DP-SGD
    global_net_dp, global_optimizer_dp, trainloader_dp = privacy_engine.make_private(
        module=global_net,
        optimizer=global_optimizer,
        data_loader=trainloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    # Snapshot the global weights at the start of the round for the proximal term
    global_params_on_device = {
        k: v.clone().to(DEVICE) for k, v in global_net.state_dict().items()
    }

    global_net_dp.train()
    local_net.train()

    for _ in range(epochs):
        for images, labels in trainloader_dp:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 1. Global model update (DP-SGD, no proximal term)
            global_optimizer_dp.zero_grad()
            global_loss = criterion(global_net_dp(images), labels)
            global_loss.backward()
            global_optimizer_dp.step()

            # 2. Local model update (SGD + Ditto proximal term)
            #    Objective: min_v  L(v) + mu/2 * ||v - w_t||^2
            local_optimizer.zero_grad()
            local_loss = criterion(local_net(images), labels)
            proximal_term = sum(
                torch.sum((p - global_params_on_device[n]) ** 2)
                for n, p in local_net.named_parameters()
                if n in global_params_on_device
            )
            total_local_loss = local_loss + (mu / 2.0) * proximal_term
            total_local_loss.backward()
            local_optimizer.step()

    # Unwrap DP module to recover clean weights
    global_net.load_state_dict(global_net_dp._module.state_dict())

    # Privacy budget consumed this round
    epsilon = privacy_engine.get_epsilon(delta=dp_delta)
    return global_net, local_net, epsilon


# ── Evaluation ────────────────────────────────────────────────────────────────
def test(net, testloader, device=None):
    """Evaluate the model and return (loss, accuracy)."""
    if device is None:
        device = DEVICE
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss    += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.max(1)[1] == labels).sum().item()
    loss     /= len(testloader.dataset)
    accuracy  = correct / len(testloader.dataset)
    return loss, accuracy


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Hardware computing initialized on: {DEVICE}")
    global_model = load_model()
    local_model  = load_model()
    local_model.load_state_dict(global_model.state_dict())

    trainloader, testloader = load_data()

    print("Starting Ditto + DP training...")
    global_model, local_model, _ = train_ditto_dp(
        global_model, local_model, trainloader, epochs=2
    )

    print("Starting evaluation of local model...")
    loss, accuracy = test(local_model, testloader)
    print(f"Final test loss:     {loss:.4f}")
    print(f"Final test accuracy: {accuracy:.4f}")
