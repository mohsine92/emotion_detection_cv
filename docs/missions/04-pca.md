# Mission 04 — Réduction PCA

Branche : `mission/04-pca`

Responsable : à attribuer dans l'issue GitHub.

Fichier : `src/emotion_detection/pca.py`

Dépend de : 03.

Livrable : pca, X_train_pca, X_val_pca, X_test_pca.

## Tâches et critères de réussite

- [ ] Ajuster la PCA uniquement sur l'entraînement et transformer validation/test.
- [ ] Étudier 100 à 200 composantes et tracer la variance expliquée cumulée.
- [ ] Justifier le choix et conserver le même objet PCA pour toutes les transformations.
- [ ] Coordonner avec la mission 05 : pour une validation croisée stricte, ajuster la PCA dans chaque pli via un Pipeline sklearn ; documenter tout écart au schéma pédagogique.
- [ ] Documenter comment exécuter et vérifier la contribution.
- [ ] Faire relire puis intégrer la Pull Request dans main.

Lire [les contrats communs](../CONTRATS.md) et [le guide de contribution](../../CONTRIBUTING.md).
