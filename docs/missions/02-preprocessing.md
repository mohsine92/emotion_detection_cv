# Mission 02 — Prétraitement PyTorch

Branche : `mission/02-preprocessing`

Responsable : à attribuer dans l'issue GitHub.

Fichier : `src/emotion_detection/preprocessing.py`

Dépend de : 01.

Livrable : train_ds, val_ds, test_ds, class_to_idx.

## Tâches et critères de réussite

- [ ] Appliquer Grayscale, Resize((48, 48)) et ToTensor.
- [ ] Documenter le périmètre et le découpage 70/15/15 proposé dans le sujet ; ne pas mélanger silencieusement les splits existants.
- [ ] Fixer la graine et vérifier que les sous-ensembles ne se chevauchent pas.
- [ ] Afficher des exemples et transmettre une correspondance des classes commune.
- [ ] Documenter comment exécuter et vérifier la contribution.
- [ ] Faire relire puis intégrer la Pull Request dans main.

Lire [les contrats communs](../CONTRATS.md) et [le guide de contribution](../../CONTRIBUTING.md).
