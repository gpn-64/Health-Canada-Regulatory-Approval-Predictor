# AFT continu (lognormal) vs régresseur — durée de revue

_Généré le 2026-09-08T18:05:41+00:00_

## Verdict

**AFT (lognormal) ne récupère pas la MAE médiane.** Holdout temporel : MAE médiane 21.1 j (régresseur) vs 39.2 j (AFT), Δ = +18.1 j. Couverture p10–p90 globale 74% vs 89% (Δ = +15%) ; >700 j 0% vs 0%.

L'AFT continu ne quantifie pas le temps (contrairement au modèle discret à bacs de 30 j), mais reste **linéaire dans les covariables** : il lisse le pic de ~43 % des dossiers qui concluent pile au standard de service de 300 j, que le régresseur boosté reproduit exactement. La perte de MAE médiane n'est donc pas un artefact de bacs — c'est le coût de quitter la flexibilité de XGBoost. Ce que le cadre de survie apporte de façon constante : une meilleure couverture d'intervalle sous décalage temporel (les dossiers en cours restent censurés au lieu d'être supprimés).

## 1. Même découpage (graine 42)

| | MAE | MAE médiane | RMSE | Couv. p10–p90 | Couv. >700 j | C de Harrell |
|---|---|---|---|---|---|---|
| Régresseur | 77.0 | 19.1 | 246 | 83% | 11% | 0.732 |
| AFT lognormal | 92.5 | 42.2 | 250 | 86% | 11% | 0.739 |

## 5. Holdout temporel (décisif)

Entraînement sur les dossiers acceptés ≤2022 (pseudo-snapshot 2022-12-31) ; évaluation sur 412 dossiers acceptés 2023+. Le régresseur perd 128 dossiers encore ouverts ; l'AFT les garde censurés (128).

| | MAE | MAE médiane | Couv. p10–p90 | Couv. >700 j |
|---|---|---|---|---|
| Régresseur | 76.0 | 21.1 | 74% | 0% |
| AFT lognormal | 75.4 | 39.2 | 89% | 0% |
