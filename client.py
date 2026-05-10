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

trainloader, testloader = load_data()

class FlowerClient(NumPyClient):
    def __init__(self):
        # Ditto utilise deux modèles : un global (pour le serveur) et un local (personnalisé)
        self.global_net = load_model()
        self.local_net = load_model()
        self.local_net.load_state_dict(self.global_net.state_dict())
        
        from opacus import PrivacyEngine
        self.privacy_engine = PrivacyEngine()  # persistante entre les rounds
        self.total_epsilon = 0.0

    def get_parameters(self, config):
        # On renvoie les paramètres du modèle global
        return [val.cpu().numpy() for _, val in self.global_net.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.global_net.state_dict()
        param_dict = zip(state_dict.keys(), parameters)
        update_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        state_dict.update(update_dict)
        self.global_net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config): 
        start_time = time.time()
        tracemalloc.start()
        
        # 1. Réception des poids globaux w_t
        self.set_parameters(parameters)
        
        # FIX 1 : synchroniser local_net sur les poids globaux reçus avant chaque round
        self.local_net.load_state_dict(self.global_net.state_dict())
        
        # 2. Entraînement Ditto + Local DP
        mu = config.get("proximal_mu", 0.01) # Si le serveur l'envoie via la configuration
        self.global_net, self.local_net, dp_epsilon = train_ditto_dp(self.global_net, self.local_net, trainloader, epochs=1, mu=mu)
        
        # Accumulation de l'epsilon
        self.total_epsilon += dp_epsilon
        
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
        
        return sparse_params_to_return, len(trainloader.dataset), metrics
    
    def evaluate(self, parameters, config): 
        start_time = time.time()
        
        # Mise à jour avec les poids du serveur pour l'évaluation globale
        self.set_parameters(parameters)
        
        # ÉVALUATION 1 : Modèle "Global"
        # Permet de voir la précision si l'on s'arrêtait à la simple agrégation du serveur.
        _, acc_global = test(self.global_net, testloader)
        
        # ÉVALUATION 2 : Modèle Local personnalisé (Après l'entraînement Ditto)
        _, acc_local_fp32 = test(self.local_net, testloader)
        
        # ÉVALUATION 3 : Modèle Local Quantifié (Déploiement TinyML virtuel)
        net_cpu = self.local_net.to('cpu')
        net_quantized = torch.quantization.quantize_dynamic(
            net_cpu, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        loss_quantized, acc_quantized = test(net_quantized, testloader, device=torch.device('cpu'))
        
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
        
        return float(loss_quantized), len(testloader.dataset), metrics

# Entry point for Flower
def client_fn(context: Context):
    return FlowerClient().to_client()

app = ClientApp(client_fn=client_fn)
