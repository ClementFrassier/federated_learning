# Indicateurs de Performance Fédérée (KPIs) — Nurse Stress FL

Ce fichier liste les métriques clés comparées entre le modèle fédéré (FL)
et le baseline centralisé pour la détection de stress des infirmières.

## Dataset & Architecture

| Paramètre | Valeur |
|---|---|
| Dataset | Nurse Stress Dataset (Empatica E4) |
| Signaux | EDA, HR, TEMP, X, Y, Z (accéléromètre) |
| Clients FL | 15 infirmières (partition naturelle par `id`) |
| Modèle | **MLP Tiny** : FC(24→32)→ReLU→FC(32→16)→ReLU→FC(16→2) |
| Paramètres | ~1 362 |
| Taille FP32 | ~5.4 KB |
| Taille INT8 | ~1.4 KB |
| Fenêtre | 60 échantillons, pas 30 (50% overlap) |
| Features | [mean, std, min, max] × 6 signaux = **24 features** |
| Label | 0 = non stressé, 1 = stressé |

---

## 1. Métriques de Qualité d'Apprentissage

- **Test Loss** : Perte moyenne sur les données de test (minimiser).
- **accuracy (INT8)** : Précision du modèle quantisé INT8 — métrique principale TinyML.
- **acc_global (FP32)** : Précision du modèle global reçu du serveur (sans adaptation locale).
- **acc_local_fp32** : Précision du modèle local FP32 après entraînement local.

---

## 2. Équité et Robustesse (Fairness)

- **accuracy_stddev** : Écart-type de la précision INT8 entre les 15 infirmières.
  → Une valeur faible prouve que le modèle est juste pour **toutes** les nurses, pas seulement celles avec le plus de données.

---

## 3. Confidentialité et Sécurité (Médical)

- **dp_epsilon (ε)** : Budget de confidentialité différentielle consommé.
  → Cible médicale recommandée : **ε ≤ 1** tout en maintenant une précision acceptable.
  → Accumulé correctement car le `PrivacyEngine` est persistant entre les rounds.

---

## 4. Personnalisation

- **local_vs_global_gap** : Différence `acc_local_fp32 − acc_global`.
  → Un gap **positif** prouve que l'entraînement local FedProx apporte une valeur par rapport au modèle global brut.

---

## 5. Comparaison FL vs Centralisé

| Métrique | FL (`results_nurse_mlp_tiny.csv`) | Centralisé (`results_baseline_centralized.csv`) |
|---|---|---|
| accuracy INT8 | ✅ privé, distribué | référence (avantage déloyal) |
| acc_fp32 | FL | centralisé |
| train_time_s | par round × rounds | total |
| dp_epsilon | accumulé | N/A (pas de DP) |
| model_size_kb | ~1.4 KB INT8 | ~1.4 KB INT8 |

---

## 6. Déploiement Embarqué (TinyML)

- **quantization_error** : Chute d'accuracy FP32 → INT8.
  → Cible : **< 2 %** pour un déploiement fiable sur MCU.
- **Peak RAM (MB)** : Pic mémoire vive côté client pendant l'entraînement local.
- **Model Size (KB)** : Taille du modèle (cible : **< 10 KB** pour MCU type STM32/ESP32).
- **Comm Size (MB)** : Poids des mises à jour envoyées au serveur par round.
  → Avec ~1 362 paramètres FP32 : **~5.2 KB** par round.

---

## 7. Énergie et Vitesse

- **Estimated Energy** : Heuristique embarquée : `2.0 × fit_time + 15.0 × comm_size_mb`
  (la radio coûte ~7.5× plus cher que le CPU).
- **fit_time (s)** : Temps moyen d'entraînement local par client par round.
- **eval_time (s)** : Temps d'évaluation (inclut la quantification INT8 dynamique).

---

*Les résultats FL sont automatiquement enregistrés dans `resultsfeat/results_nurse_mlp_tiny.csv`.*
*Les résultats centralisés sont dans `resultsfeat/results_baseline_centralized.csv`.*
