# Mission 05 — Régression logistique et GridSearchCV

Branche : `mission/05-logreg`

Responsable : à attribuer dans l'issue GitHub.

Fichier : `src/emotion_detection/training.py`

Dépend de : 04.

Livrable : clf, paramètres retenus et score de validation.

## Tâches et critères de réussite

- [ ] Utiliser la régression logistique multinomiale, lbfgs et L2.
- [ ] Comparer C = 0.1, 1, 3, 10 avec GridSearchCV, cv=3.
- [ ] Documenter la convergence et la compatibilité avec la version de scikit-learn.
- [ ] Présenter les scores de validation sans utiliser le test pour choisir les paramètres.
- [ ] Accorder avec la mission 04 si clf reçoit des pixels via un Pipeline ou des données PCA ; éviter une double transformation.
- [ ] Documenter comment exécuter et vérifier la contribution.
- [ ] Faire relire puis intégrer la Pull Request dans main.

Lire [les contrats communs](../CONTRATS.md) et [le guide de contribution](../../CONTRIBUTING.md).
