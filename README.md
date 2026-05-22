# When Machine Learning Fails — Mini-Project ECC Spring 2026
## Diagnosing and Repairing Class Imbalance Failure on the Online Shoppers Dataset

---

## Objectif du Projet

Investiguer scientifiquement le **mode d'échec "Class Imbalance"** sur le dataset UCI Online Shoppers Purchasing Intention (UCI 468). Ce projet suit une démarche de recherche rigoureuse : symptôme → hypothèse causale → expérience contrôlée → correction ciblée.

**Question de recherche :**
 Le déséquilibre de classes (85% non-achat / 15% achat) cause-t-il aux classificateurs non-linéaires d'atteindre une accuracy globale élevée tout en échouant à détecter correctement les acheteurs (`Revenue=True`) ?

---

##  Dataset

| Propriété | Valeur |
|-----------|--------|
| Source | UCI Machine Learning Repository (ID: 468) |
| Fichier | `online_shoppers_intention.csv` |
| Observations | 12 330 sessions web |
| Features | 17 (numériques + catégorielles) |
| Cible | `Revenue` (True = achat, False = non-achat) |
| Déséquilibre | 84.5% False / 15.5% True |

---

##  Mode d'Échec Étudié : Class Imbalance

**Symptôme** : Accuracy élevée (>85%) mais Recall très faible pour `Revenue=True`.  
**Cause** : Les modèles apprennent une frontière biaisée vers la classe majoritaire.  
**Test** : Dégradation artificielle du ratio de positifs → vérification que le recall suit.  
**Correction** : SMOTE (rééchantillonnage) + AdaBoost (boosting adaptatif).

---

##  Ordre des Notebooks

| # | Fichier | Objectif |
|---|---------|----------|
| 1 | `01_Untouched_class_imbalance_NB1.ipynb` | Exposer le symptôme d'échec avec RF, MLP, LightGBM sans correction |
| 2 | `02_SMOTE_AdaBoost_Classification_Imbalance_Correction.ipynb` | Corriger avec SMOTE + AdaBoost + expérience contrôlée |
| 3 | `03_PSO_SMOTE_AdaBoost_FeatureSelection_Imbalance_Correction.ipynb` | Tester PSO + SMOTE + AdaBoost (sélection de features) |

---

## ▶ Comment Exécuter

### Sur Google Colab
1. Uploader `online_shoppers_intention.csv` dans l'environnement Colab (panneau Files > Upload)
2. Ouvrir chaque notebook dans l'ordre (NB1 → NB2 → NB3)
3. Exécuter toutes les cellules (`Runtime > Run all`)
4. NB2 et NB3 peuvent charger les résultats CSV générés par NB1/NB2

### Packages installés automatiquement
```bash
pip install imbalanced-learn lightgbm pyswarms
```

---

## 📦 Outputs Attendus

### Notebook 1
- `notebook1_untouched_results.csv` — Métriques validation + test
- `eda_overview.png` — Vue d'ensemble EDA
- `skewness_analysis.png` — Analyse log1p
- `notebook1_final_evaluation.png` — Matrice + ROC + PR

### Notebook 2
- `notebook2_smote_adaboost_results.csv` — Résultats grid search
- `notebook2_final_evaluation.png` — Matrice + ROC + PR
- `notebook2_degradation_experiment.png` — Expérience contrôlée

### Notebook 3
- `notebook3_pso_smote_adaboost_results.csv` — Résultats PSO + AdaBoost
- `notebook3_pso_convergence.png` — Convergence PSO (si manuel)
- `notebook3_final_evaluation.png` — Matrice + ROC + PR
- `notebook3_global_comparison.png` — Comparaison 3 pipelines
- `notebook3_feature_reduction.png` — Réduction de complexité

---

##  Règles Anti-Fuite (Data Leakage Prevention)

| Règle | Détail |
|-------|--------|
| Split d'abord | Train (60%) / Val (20%) / Test (20%) **avant tout prétraitement** |
| Scaler/Encoder | Fittés **uniquement sur X_train**, transforment Val et Test |
| SMOTE | Appliqué **uniquement sur X_train**, jamais avant le split |
| Sélection PSO | Évaluée sur un **inner split de X_train** uniquement |
| Hyperparamètres | Tuned sur **Val uniquement** |
| Test set | Utilisé **une seule fois** pour l'évaluation finale |

---

##  Métriques Prioritaires

1. **Recall pour Revenue=True** — Mesure la proportion d'acheteurs correctement identifiés
2. **F1-score pour Revenue=True** — Équilibre précision/recall
3. **PR-AUC** — Performance globale sur la classe minoritaire
4. ROC-AUC, Accuracy (métriques complémentaires)

L'Accuracy seule est trompeuse en contexte de déséquilibre de classes.

---

## Reproductibilité

```python
random_state = 42
np.random.seed(42)
```

Toutes les fonctions stochastiques utilisent `random_state=42`.

---

##  Équipe 

- **Equipe** : SAAIDI Salma & EL BAHTOURI Aya