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
    """CNN for FashionMNIST (1-channel, 28×28).

    GroupNorm layers replace BatchNorm.
    """

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.gn1   = nn.GroupNorm(num_groups=2, num_channels=6)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.gn2   = nn.GroupNorm(num_groups=4, num_channels=16)
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
def load_data(
    node_id: int = 0,
    num_clients: int = 10,
    batch_size: int = 32,
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_model_size(model) -> float:
    """Return model size in megabytes (parameters + buffers)."""
    param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


# ── Ditto Training ────────────────────────────────────────────────────────────
def train_ditto(
    global_net,
    global_optimizer,
    trainloader,
    local_net,
    epochs: int,
    mu: float = 0.05,
    lr_local: float = 0.005,
    momentum_local: float = 0.9,
):
    """One federation round of Ditto training.

    The global model is trained with standard SGD.
    The local (personalised) model is trained with standard SGD and a
    proximal penalty that keeps it anchored to the incoming global weights.
    """
    criterion      = torch.nn.CrossEntropyLoss()
    local_optimizer = torch.optim.SGD(
        local_net.parameters(), lr=lr_local, momentum=momentum_local
    )

    # Snapshot global weights at round start for the Ditto proximal term
    global_params_on_device = {
        k: v.clone().to(DEVICE)
        for k, v in global_net.state_dict().items()
    }

    global_net.train()
    local_net.train()

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 1. Global model update (no proximal term)
            global_optimizer.zero_grad()
            global_loss = criterion(global_net(images), labels)
            global_loss.backward()
            global_optimizer.step()

            # 2. Local model update (SGD + Ditto proximal term)
            local_optimizer.zero_grad()
            local_loss = criterion(local_net(images), labels)
            proximal_term = sum(
                torch.sum((p - global_params_on_device[n]) ** 2)
                for n, p in local_net.named_parameters()
                if n in global_params_on_device
            )
            (local_loss + (mu / 2.0) * proximal_term).backward()
            local_optimizer.step()

    return global_net, local_net


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
            loss   += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.max(1)[1] == labels).sum().item()
    loss    /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    global_model = load_model()
    local_model  = load_model()
    local_model.load_state_dict(global_model.state_dict())

    trainloader, testloader = load_data()

    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=0.01, momentum=0.9)

    print("Starting Ditto training...")
    global_model, local_model = train_ditto(
        global_model, global_optimizer, trainloader,
        local_model, epochs=2, mu=0.05,
    )

    print("Evaluating local model...")
    loss, acc = test(local_model, testloader)
    print(f"Loss: {loss:.4f}  |  Accuracy: {acc:.4f}")
