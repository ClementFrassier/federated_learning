# Indicateurs de Performance Fédérée (KPIs)

Ce fichier liste les métriques clés à comparer entre les différentes architectures et branches Git de ce projet.

## 1. Métriques de Qualité d'Apprentissage
- **Test Loss** : La perte moyenne sur les données de test (doit être la plus basse possible).
- **Test Accuracy** : La précision globale du modèle sur les données de test (doit être la plus haute possible).

## 2. Équité et Robustesse (Fairness)
- **Accuracy StdDev** : L'écart-type de la précision entre les différents clients. Une faible valeur prouve que l'algorithme est juste et n'oublie aucun "patient".

## 3. Confidentialité et Sécurité (Médical)
- **DP Epsilon ($\epsilon$)** : Le budget de confidentialité consommé pendant l'entraînement. Un bon algorithme doit maintenir $\epsilon \leq 1$ tout en gardant une bonne précision.

## 4. Personnalisation (Ditto)
- **Local vs Global Gap** : La différence de précision entre le modèle personnalisé (Local FP32) et le modèle généralisé (Global FP32) sur les mêmes données patient. Un gap positif prouve l'efficacité de Ditto.

## 5. Déploiement Embarqué (TinyML)
- **Quantization Error** : La chute de précision (Accuracy Drop) due à la conversion des poids de `float32` vers `qint8`. Plus elle est proche de 0, mieux c'est.
- **Peak RAM (MB)** : Le pic de mémoire vive consommé par l'appareil pendant la phase d'entraînement local.
- **Model Size (MB)** : La taille des paramètres du modèle.
- **Comm Size (MB)** : Le poids des pseudo-gradients envoyés au serveur.

## 6. Énergie et Vitesse
- **Estimated Energy** : Une valeur heuristique (ex: `2.0 * Fit Time + 15.0 * Comm Size`) pour estimer la consommation de batterie de l'objet connecté (Wearable). La Radio coûte généralement plus cher que le CPU.
- **Fit Time (s)** : Le temps moyen d'entraînement local par client.
- **Eval Time (s)** : Le temps moyen d'évaluation (inclut la quantification dynamique de test).

---
*Note : Ces données sont automatiquement enregistrées dans le fichier `results.csv` à la fin de l'exécution.*
