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

    GroupNorm layers replace BatchNorm for Opacus (DP-SGD) compatibility.
    """

    def __init__(self) -> None:
        super(Net, self).__init__()
        # 1 input channel (greyscale)
        self.conv1 = nn.Conv2d(1, 6, 5)
        # GroupNorm required for Opacus compatibility
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
def load_data(node_id: int = 0, num_clients: int = 10, batch_size: int = 32,
              alpha: float = 0.3, seed: int = 42):
    """Load FashionMNIST with Dirichlet non-IID partitioning.

    Args:
        node_id:     Client index (0 … num_clients-1).
        num_clients: Total number of federated clients.
        batch_size:  Mini-batch size for returned DataLoaders.
        alpha:       Dirichlet concentration (lower = more heterogeneous).
        seed:        RNG seed for reproducible partitioning.

    Returns:
        (trainloader, testloader) — testloader uses the full global test set.
    """
    trf = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    trainset = FashionMNIST("./data", train=True,  download=True, transform=trf)
    testset  = FashionMNIST("./data", train=False, download=True, transform=trf)

    if alpha <= 0.0:
        # Fallback: uniform IID slice
        n       = len(trainset) // num_clients
        indices = list(range(node_id * n, (node_id + 1) * n))
    else:
        # Dirichlet non-IID partition
        rng     = np.random.default_rng(seed)
        labels  = np.array(trainset.targets)
        n_classes = int(labels.max()) + 1
        class_indices = [np.where(labels == c)[0] for c in range(n_classes)]
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]
        for c_idx in class_indices:
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([
                p * (len(ci) < len(trainset) / num_clients)
                for p, ci in zip(proportions, client_indices)
            ])
            proportions = proportions / proportions.sum()
            splits      = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
            for i, part in enumerate(np.split(c_idx, splits)):
                client_indices[i].extend(part.tolist())
        indices = client_indices[node_id]

    client_trainset = torch.utils.data.Subset(trainset, indices)
    trainloader = DataLoader(client_trainset, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(testset,         batch_size=batch_size, shuffle=False)
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

    Zeroes out weights below the (sparsity_ratio × 100)-th percentile,
    keeping the top (1 − sparsity_ratio) fraction by absolute value.
    """
    sparse_params = []
    for param in parameters:
        if param.size > 0:
            threshold  = np.percentile(np.abs(param), sparsity_ratio * 100)
            param_copy = param.copy()
            param_copy[np.abs(param_copy) < threshold] = 0.0
            sparse_params.append(param_copy)
        else:
            sparse_params.append(param)
    return sparse_params


# ── Ditto + Local DP Training ─────────────────────────────────────────────────
def train_ditto_dp(
    global_net_dp,
    global_optimizer_dp,
    trainloader_dp,
    local_net,
    epochs: int,
    mu: float = 0.05,
    lr_local: float = 0.001,
    momentum_local: float = 0.9,
):
    """One federation round of Ditto training.

    The global model is trained with DP-SGD (Opacus).
    The local (personalised) model is trained with standard SGD and a
    proximal penalty that keeps it anchored to the incoming global weights.

    IMPORTANT — The PrivacyEngine is NOT created here.
    It must be created once in FlowerClient.__init__ and kept alive across
    all rounds so that epsilon accumulates correctly over the full experiment.
    Creating a fresh PrivacyEngine each call would silently reset ε to 0.

    Args:
        global_net_dp:        Opacus-wrapped global model (persistent).
        global_optimizer_dp:  Opacus-wrapped optimiser (persistent).
        trainloader_dp:       Opacus-wrapped DataLoader (persistent).
        local_net:            Personalised local model (plain PyTorch).
        epochs:               Number of local training epochs per round.
        mu:                   Ditto proximal penalty weight µ.
        lr_local:             Learning rate for the local SGD optimiser.
        momentum_local:       Momentum for the local SGD optimiser.

    Returns:
        Tuple (global_net_unwrapped, local_net)
    """
    criterion       = torch.nn.CrossEntropyLoss()
    local_optimizer = torch.optim.SGD(
        local_net.parameters(), lr=lr_local, momentum=momentum_local
    )

    # Snapshot global weights at round start for the Ditto proximal term
    global_params_on_device = {
        k: v.clone().to(DEVICE)
        for k, v in global_net_dp._module.state_dict().items()
    }

    global_net_dp.train()
    local_net.train()

    for _ in range(epochs):
        for images, labels in trainloader_dp:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 1. Global model update (DP-SGD — no proximal term)
            global_optimizer_dp.zero_grad()
            global_loss = criterion(global_net_dp(images), labels)
            global_loss.backward()
            global_optimizer_dp.step()

            # 2. Local model update (SGD + Ditto proximal term)
            #    Objective: min_v L(v) + µ/2 · ||v − w_t||²
            local_optimizer.zero_grad()
            local_loss    = criterion(local_net(images), labels)
            proximal_term = sum(
                torch.sum((p - global_params_on_device[n]) ** 2)
                for n, p in local_net.named_parameters()
                if n in global_params_on_device
            )
            (local_loss + (mu / 2.0) * proximal_term).backward()
            local_optimizer.step()

    # Return the unwrapped global weights (without the Opacus Ghost wrapper)
    return global_net_dp._module, local_net


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


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    from opacus import PrivacyEngine as PE
    print(f"Device: {DEVICE}")
    global_model = load_model()
    local_model  = load_model()
    local_model.load_state_dict(global_model.state_dict())

    trainloader, testloader = load_data()

    privacy_engine   = PE()
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=0.001, momentum=0.9)
    global_model_dp, global_optimizer_dp, trainloader_dp = privacy_engine.make_private(
        module=global_model,
        optimizer=global_optimizer,
        data_loader=trainloader,
        noise_multiplier=1.5,
        max_grad_norm=1.0,
    )

    print("Starting Ditto + DP training...")
    global_model, local_model = train_ditto_dp(
        global_model_dp, global_optimizer_dp, trainloader_dp,
        local_model, epochs=2, mu=0.05,
    )

    print("Evaluating local model...")
    loss, acc = test(local_model, testloader)
    print(f"Loss: {loss:.4f}  |  Accuracy: {acc:.4f}")
