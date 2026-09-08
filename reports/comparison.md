# Comparaison régression/classification vs survie à risques concurrents

_Généré le 2026-09-08T15:12:18+00:00_

## Verdict

**RESTER sur `main` (régresseur + classifieur)**

Critère de décision : sur le holdout temporel (entraînement sur les dossiers acceptés ≤2022, évaluation sur ceux acceptés en 2023+), adopter si la couverture p10–p90 pour les revues réellement > 700 j progresse nettement au-dessus de la ligne de base (> +5 points), **sans** dégradation de la MAE médiane, et si le Brier de la probabilité d'autorisation ne se dégrade pas.

- Couverture > 700 j (holdout) : régression = 0.0, survie = 0.07142857142857142 -> amélioration nette
- MAE médiane (holdout) : régression = 21.1 j, survie = 24.5 j -> dégradée
- Brier approbation (holdout) : régression = 0.0495, survie = 0.0480 -> préservé

## 1. Même découpage (graine 42) — durée

|                                  | MAE  | MAE médiane | RMSE  |
| -------------------------------- | ---- | ------------ | ----- |
| Régresseur                      | 77.0 | 19.1         | 245.7 |
| Survie (médiane conditionnelle) | 71.5 | 30.0         | 146.9 |

## 2. Intervalle p10–p90

|             | Couverture globale | Largeur médiane | Couverture (>700j réels) | n>700j |
| ----------- | ------------------ | ---------------- | ------------------------- | ------ |
| Régresseur | 82.8%              | 237 j            | 11.1%                     | 9      |
| Survie      | 78.9%              | 120 j            | 18.2%                     | 11     |

## 3. Approbation

|                           | ROC-AUC | Brier  |
| ------------------------- | ------- | ------ |
| Classifieur               | 0.664   | 0.0762 |
| Survie (CIF autorisation) | 0.629   | 0.0788 |

## 4. Métriques natives de survie (censurés inclus)

n événements = 303, n censurés = 168

|                                       | C de Harrell | C de Uno | AUC(700j) | Brier IPCW(700j) |
| ------------------------------------- | ------------ | -------- | --------- | ---------------- |
| Régresseur (score = -jours prédits) | 0.732        | 0.708    | 0.832     | n/a              |
| Survie                                | 0.728        | 0.704    | 0.756     | 0.0675           |

## 5. Holdout temporel (test décisif)

Entraînement sur 975 dossiers acceptés ≤2022 (pseudo-snapshot 2022-12-31) ; évaluation sur 412 dossiers acceptés 2023+.

Le pipeline d'origine a supprimé 128 dossiers encore ouverts au pseudo-snapshot ; le modèle de survie les garde censurés (128 dossiers).

|                         | MAE médiane | Couverture >700j | ROC-AUC approbation | Brier approbation |
| ----------------------- | ------------ | ---------------- | ------------------- | ----------------- |
| Régresseur/classifieur | 21.1 j       | 0.0%             | 0.554               | 0.0495            |
| Survie                  | 24.5 j       | 7.1%             | 0.606               | 0.0480            |

## Note de méthode

La MAE (médiane) de la survie est mécaniquement plancherée à la demi-largeur d'un bac de 30 jours (quantification en bacs), ce qui désavantage structurellement sa MAE face au régresseur en régime non censuré. Le point décisif est le holdout temporel (section 5), la seule configuration qui reproduit le biais de troncature par snapshot que le modèle de survie est censé corriger.
