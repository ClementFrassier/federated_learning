# Federated Learning — Ditto Only (FedAvg)

Simulation d'apprentissage fédéré (FL) avec personnalisation **Ditto** sur **Flower v2 (1.29.0)**, entraînée sur le dataset **FashionMNIST**.

Branche : `feat/ditto-only`

## Architecture

Ce projet implémente l'algorithme Ditto dans sa forme épurée :

| Composant | Rôle |
|---|---|
| **Ditto** | Optimisation bi-objectif : entraînement d'un modèle global (transmis au serveur via FedAvg) et d'un modèle local personnalisé (optimisé localement sur l'appareil avec une pénalité proximale $\mu$ par rapport au modèle global). |
| **FedAvg** | Stratégie d'agrégation standard de Flower, sans optimiseur adaptatif côté serveur. |
| **Pas de DP / Pas de SecAgg** | Differential Privacy et SecAgg retirés pour servir de base de comparaison directe. |
| **Évaluation INT8** | Quantification dynamique INT8 du modèle local effectuée à l'évaluation pour mesurer les métriques TinyML. |

### Structure des fichiers

*   `task.py` : Architecture du modèle `Net` (CNN avec GroupNorm), chargement des partitions locales IID, fonction d'entraînement Ditto (`train_ditto`) et fonction d'évaluation (`test`).
*   `client_app.py` : `ClientApp` gérant le cycle de vie du client, l'évaluation à trois niveaux (global, local FP32, local INT8) et l'entraînement Ditto local.
*   `server_app.py` : `ServerApp` gérant la stratégie Flower `FedAvg` et l'exportation des métriques dans un fichier CSV.
*   `pyproject.toml` : Configuration des hyperparamètres de l'expérience et dépendances Flower.

## Lancement de la Simulation

Pour installer les dépendances nécessaires :
```bash
pip install -e .
```

Pour lancer la simulation de référence (10 clients, 10 rounds) :
```bash
flwr run . --stream
```

## Indicateurs KPI

Les résultats de la simulation sont enregistrés dans le fichier :
`resultsfeat/results_ditto-only.csv`

Les indicateurs mesurés sont identiques à ceux des autres branches pour permettre une comparaison directe :
*   **KPIs d'entraînement** (FIT) : `fit_time`, `peak_ram_mb`, `comm_size_mb`, `model_size_mb`, `dp_epsilon` (mis à `0.0`), `estimated_energy`
*   **KPIs d'évaluation** (EVAL) : `accuracy` (INT8), `acc_global`, `acc_local_fp32`, `local_vs_global_gap`, `quantization_error`, `accuracy_stddev`, `quantized_model_size_mb`, `eval_time`, `loss`
