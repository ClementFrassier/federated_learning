import warnings
warnings.filterwarnings("ignore")

from centralized import load_data, load_model, train_fedprox_dp, test, get_model_size

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
        # Un seul modèle. FedGN garde les couches de normalisation en local.
        self.net = load_model()
        self.net_before_fit = load_model()
        self.net_before_fit.load_state_dict(self.net.state_dict())
        # Chaque client charge SA partition
        self.trainloader, self.testloader = load_data(
            node_id=node_id, num_clients=10, batch_size=64
        )

    def get_parameters(self, config):
        # On ne renvoie pas les paramètres GroupNorm (FedGN) au serveur
        return [val.cpu().numpy() for name, val in self.net.state_dict().items() if "gn" not in name]

    def set_parameters(self, parameters):
        state_dict = self.net.state_dict()
        # On ne met à jour que les paramètres qui ne sont pas GroupNorm
        keys_to_update = [k for k in state_dict.keys() if "gn" not in k]
        param_dict = zip(keys_to_update, parameters)
        update_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        state_dict.update(update_dict)
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config): 
        start_time = time.time()
        tracemalloc.start()
        
        # 1. Sauvegarde du modèle avant la mise à jour (pour l'évaluation du "Global Gap" futur)
        self.net_before_fit = load_model()
        self.net_before_fit.load_state_dict(self.net.state_dict())
        
        # 2. Réception des poids globaux w_t (met à jour Conv/Linear, garde GN localement)
        self.set_parameters(parameters)
        
        # On extrait le dictionnaire des poids globaux pour la pénalité FedProx
        global_params_dict = {k: v for k, v in self.net.state_dict().items() if "gn" not in k}
        
        # 3. Entraînement FedProx + Local DP
        mu = config.get("proximal_mu", 0.1) # Si le serveur l'envoie via la configuration
        self.net, dp_epsilon = train_fedprox_dp(self.net, global_params_dict, self.trainloader, epochs=1, mu=mu)
        
        # On ne renvoie pas les couches de normalisation (FedGN)
        params_to_return = self.get_parameters(config)
        
        # Calcul de la taille de communication de ce que l'on envoie au serveur
        comm_size_mb = sum([p.nbytes for p in params_to_return]) / (1024 * 1024)
        
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
            "model_size_mb": float(get_model_size(self.net)),
            "dp_epsilon": float(dp_epsilon),
            "estimated_energy": float(estimated_energy)
        }
        
        return params_to_return, len(self.trainloader.dataset), metrics
    
    def evaluate(self, parameters, config): 
        start_time = time.time()
        
        # Mise à jour avec les poids du serveur pour l'évaluation globale
        self.set_parameters(parameters)
        
        # ÉVALUATION 1 : Modèle "Global" (Utilise les poids du serveur + les stats GN de l'avant-dernier round)
        # Permet de voir la précision si l'on s'arrêtait à la simple agrégation du serveur.
        _, acc_global = test(self.net_before_fit, self.testloader)
        
        # ÉVALUATION 2 : Modèle Local personnalisé (Après l'entraînement FedProx + FedGN)
        _, acc_local_fp32 = test(self.net, self.testloader)
        
        # ÉVALUATION 3 : Modèle Local Quantifié (Déploiement TinyML virtuel)
        # Copie propre sur CPU pour ne pas déplacer self.net du GPU
        net_cpu = load_model().cpu()
        net_cpu.load_state_dict({k: v.cpu() for k, v in self.net.state_dict().items()})
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
