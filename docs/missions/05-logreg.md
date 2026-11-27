# Mission 05 — Régression logistique et GridSearchCV

Branche : `mission/05-logreg`

Responsable : Mohsine

Fichier : `src/emotion_detection/training.py`

Dépend de : 04.

Livrable : clf, paramètres retenus et score de validation.

## Tâches et critères de réussite

- [x] Utiliser la régression logistique multinomiale, lbfgs et L2.
- [x] Comparer C = 0.1, 1, 3, 10 avec GridSearchCV, cv=3.
- [x] Documenter la convergence et la compatibilité avec la version de scikit-learn.
- [x] Présenter les scores de validation sans utiliser le test pour choisir les paramètres.
- [x] Accorder avec la mission 04 si clf reçoit des pixels via un Pipeline ou des données PCA ; éviter une double transformation.
- [x] Documenter comment exécuter et vérifier la contribution.
- [ ] Faire relire puis intégrer la Pull Request dans main.

Lire [les contrats communs](../CONTRATS.md) et [le guide de contribution](../../CONTRIBUTING.md).
