import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    """CNN for FashionMNIST (1-channel, 28x28).
    FedBN: BatchNorm layers (weight, bias, and their running_mean/running_var
    statistics) stay local to each client and are never sent to the server.
    """

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        # BatchNorm: local to each client, never aggregated by the server
        self.bn1   = nn.BatchNorm2d(6)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2   = nn.BatchNorm2d(16)
        # FashionMNIST 28x28 -> after 2x(conv5 + pool2) -> 4x4
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_data(
    node_id: int = 0,
    num_clients: int = 10,
    batch_size: int = 64,
    alpha: float = 0.0,
    seed: int = 42,
):
    """Load FashionMNIST and return a partition for a given client.

    Partitioning strategy
    ---------------------
    alpha == 0  →  IID sequential split (original behaviour).
    alpha  > 0  →  Non-IID Dirichlet(alpha) split.
                   Lower alpha → more heterogeneous (0.1 = very skewed,
                   0.5 = moderately skewed, 1.0 ≈ IID).
    """
    trf = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    trainset = FashionMNIST("./data", train=True,  download=True, transform=trf)
    testset  = FashionMNIST("./data", train=False, download=True, transform=trf)

    if alpha > 0.0:
        # ── Dirichlet non-IID ──────────────────────────────────────────────────
        rng = np.random.default_rng(seed)
        targets = np.array(trainset.targets)
        num_classes = int(targets.max()) + 1
        client_indices = [[] for _ in range(num_clients)]

        for cls in range(num_classes):
            cls_idx = np.where(targets == cls)[0]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet([alpha] * num_clients)
            counts = (proportions * len(cls_idx)).astype(int)
            # Fix rounding so all samples are assigned
            counts[-1] = len(cls_idx) - counts[:-1].sum()
            ptr = 0
            for cid, n in enumerate(counts):
                client_indices[cid].extend(cls_idx[ptr: ptr + n].tolist())
                ptr += n

        indices = client_indices[node_id]
    else:
        # ── IID sequential (original) ──────────────────────────────────────────
        n = len(trainset) // num_clients
        indices = list(range(node_id * n, (node_id + 1) * n))

    client_trainset = torch.utils.data.Subset(trainset, indices)
    trainloader = DataLoader(client_trainset, batch_size=batch_size, shuffle=True)

    # ── Global testloader (full 10 k images — measures generalisation) ────────
    testloader_global = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # ── Local testloader (same class distribution as trainset) ────────────────
    # For personalisation-aware algorithms (Ditto) the local model should be
    # evaluated on data that matches the client's own distribution.
    # We sample test images proportionally to the class frequencies in trainset.
    train_targets = np.array([trainset.targets[i] for i in indices])
    test_targets  = np.array(testset.targets)
    num_classes   = int(test_targets.max()) + 1

    local_test_indices: list = []
    for cls in range(num_classes):
        cls_count = int((train_targets == cls).sum())
        if cls_count == 0:
            continue
        cls_test_idx = np.where(test_targets == cls)[0]
        # Sample proportionally (capped at available test images for that class)
        n_local = min(cls_count, len(cls_test_idx))
        rng_test = np.random.default_rng(seed + node_id + cls)   # deterministic
        chosen = rng_test.choice(cls_test_idx, size=n_local, replace=False)
        local_test_indices.extend(chosen.tolist())

    local_testset  = torch.utils.data.Subset(testset, local_test_indices)
    testloader_local = DataLoader(local_testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader_global, testloader_local


def load_model() -> Net:
    """Instantiate a fresh Net and move it to the available device."""
    return Net().to(DEVICE)


def get_model_size(model) -> float:
    """Return model size in megabytes (parameters + buffers)."""
    param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 ** 2


# ── Standard local training ───────────────────────────────────────────────────
def train(net, trainloader, epochs: int, lr: float = 0.01, momentum: float = 0.9):
    """Train the model with standard SGD (no DP, no FedProx)."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


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
