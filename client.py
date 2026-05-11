import warnings
warnings.filterwarnings("ignore")

from centralized import load_data, load_model, train_ditto_dp, apply_sparsification, test, get_model_size

from collections import OrderedDict
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import torch
import torch.quantization
import numpy as np
import time
import tracemalloc

class FlowerClient(NumPyClient):
    def __init__(self, node_id: int):
        # Ditto utilise deux modèles : un global (pour le serveur) et un local (personnalisé)
        self.global_net = load_model()
        self.local_net = load_model()
        self.local_net.load_state_dict(self.global_net.state_dict())
        
        # Chaque client charge SA partition
        self.trainloader, self.testloader = load_data(
            node_id=node_id,
            num_clients=10,
            batch_size=128
        )
        
        from opacus import PrivacyEngine
        self.privacy_engine = PrivacyEngine()
        global_optimizer = torch.optim.SGD(self.global_net.parameters(), lr=0.001, momentum=0.9)
        self.global_net_dp, self.global_optimizer_dp, self.trainloader_dp = self.privacy_engine.make_private(
            module=self.global_net,
            optimizer=global_optimizer,
            data_loader=self.trainloader,
            noise_multiplier=0.5,
            max_grad_norm=1.0,
        )
        self.total_epsilon = 0.0

    def get_parameters(self, config):
        # Lire depuis le module unwrappé
        return [val.cpu().numpy() for _, val in self.global_net_dp._module.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.global_net_dp._module.state_dict()
        param_dict = zip(state_dict.keys(), parameters)
        update_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        state_dict.update(update_dict)
        self.global_net_dp._module.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config): 
        start_time = time.time()
        tracemalloc.start()
        
        # 1. Réception des poids globaux w_t
        self.set_parameters(parameters)
        
        # FIX 1 : synchroniser local_net sur les poids globaux reçus avant chaque round
        self.local_net.load_state_dict(self.global_net_dp._module.state_dict())
        
        # 2. Entraînement Ditto + Local DP
        mu = config.get("proximal_mu", 0.01) # Si le serveur l'envoie via la configuration
        self.global_net, self.local_net = train_ditto_dp(
            self.global_net_dp, self.global_optimizer_dp, self.trainloader_dp,
            self.local_net, epochs=1, mu=mu
        )
        self.total_epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        
        # Accumulation de l'epsilon (déjà géré par opacus get_epsilon)
        
        # 3. Sparsification des paramètres avant envoi (50% de pruning)
        params_to_return = self.get_parameters(config)
        sparsity_ratio = 0.5
        sparse_params_to_return = apply_sparsification(params_to_return, sparsity_ratio=sparsity_ratio)
        
        # Calcul de la taille de communication (fix cosmétique)
        comm_size_mb = sum([p.nbytes for p in sparse_params_to_return]) / (1024 * 1024)
        
        _, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        fit_time = time.time() - start_time
        peak_ram_mb = peak_ram / (1024 * 1024)
        
        # Energy Proxy : CPU Alpha = 2.0, Radio Beta = 15.0
        estimated_energy = (2.0 * fit_time) + (15.0 * comm_size_mb)
        
        metrics = {
            "fit_time": float(fit_time),
            "peak_ram_mb": float(peak_ram_mb),
            "comm_size_mb": float(comm_size_mb),
            "model_size_mb": float(get_model_size(self.local_net)),
            "dp_epsilon": float(self.total_epsilon),
            "estimated_energy": float(estimated_energy)
        }
        
        return sparse_params_to_return, len(self.trainloader.dataset), metrics
    
    def evaluate(self, parameters, config): 
        start_time = time.time()
        
        # Mise à jour avec les poids du serveur pour l'évaluation globale
        self.set_parameters(parameters)
        
        # ÉVALUATION 1 : Modèle "Global"
        # Permet de voir la précision si l'on s'arrêtait à la simple agrégation du serveur.
        _, acc_global = test(self.global_net_dp._module, self.testloader)
        
        # ÉVALUATION 2 : Modèle Local personnalisé (Après l'entraînement Ditto)
        _, acc_local_fp32 = test(self.local_net, self.testloader)
        
        # ÉVALUATION 3 : Modèle Local Quantifié (Déploiement TinyML virtuel)
        net_cpu = type(self.local_net)()  # nouvelle instance vide
        net_cpu.load_state_dict(
            {k: v.cpu() for k, v in self.local_net.state_dict().items()}
        )

        net_quantized = torch.quantization.quantize_dynamic(
            net_cpu, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        loss_quantized, acc_quantized = test(net_quantized, self.testloader, device=torch.device('cpu'))
        
        eval_time = time.time() - start_time
        
        # Calcul des Gaps
        local_vs_global_gap = acc_local_fp32 - acc_global
        quantization_error = acc_local_fp32 - acc_quantized
        
        metrics = {
            "accuracy": float(acc_quantized),
            "acc_global": float(acc_global),
            "acc_local_fp32": float(acc_local_fp32),
            "local_vs_global_gap": float(local_vs_global_gap),
            "quantization_error": float(quantization_error),
            "eval_time": float(eval_time),
            "quantized_model_size_mb": float(get_model_size(net_quantized))
        }
        
        return float(loss_quantized), len(self.testloader.dataset), metrics

# Entry point for Flower
def client_fn(context: Context):
    node_id = context.node_config["partition-id"]
    return FlowerClient(node_id=node_id).to_client()

app = ClientApp(client_fn=client_fn)
