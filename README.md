# emotion_detection_cv

Projet B3 : classification des expressions faciales sur FER2013 avec PyTorch pour le prétraitement, puis PCA et régression logistique scikit-learn.

## Organisation

Nous travaillons d'abord dans GitHub avec des fichiers Python. Le notebook Google Colab sera assemblé plus tard pour le rendu.
Le dépôt contient pour le moment un **socle d'organisation**, pas un modèle implémenté.

- `main` : travail commun relu et intégré.
- Une branche par mission, créée depuis le même socle.
- Une issue par mission : choisir un responsable dans les Assignees.
- Une Pull Request vers `main` pour intégrer chaque contribution.
- Une relecture par un autre membre avant fusion (règle d'équipe ; protection technique à configurer séparément).
- Après chaque fusion, synchroniser les autres branches avec `main`.

| Mission | Branche | Fichier principal |
| --- | --- | --- |
| 01 — Environnement et données | `mission/01-setup-kaggle` | `src/emotion_detection/setup_data.py` |
| 02 — Prétraitement PyTorch | `mission/02-preprocessing` | `src/emotion_detection/preprocessing.py` |
| 03 — Conversion vers NumPy | `mission/03-conversion-numpy` | `src/emotion_detection/conversion.py` |
| 04 — Réduction PCA | `mission/04-pca` | `src/emotion_detection/pca.py` |
| 05 — Régression logistique et GridSearchCV | `mission/05-logreg` | `src/emotion_detection/training.py` |
| 06 — Évaluation et conclusion | `mission/06-evaluation` | `src/emotion_detection/evaluation.py` |

## Structure

```text
src/emotion_detection/   Modules Python, un par mission
tests/                  Tests rapides sans données Kaggle
docs/missions/          Fiches de travail et critères de réussite
docs/CONTRATS.md         Entrées/sorties communes
docs/ORGANISATION.md     Répartition et suivi
.github/                Modèles de tickets/PR et vérifications automatiques
```

Les dossiers locaux `data/`, `models/` et `outputs/` sont ignorés par Git.
Ne jamais publier `kaggle.json`, un token, un environnement Python ou le dataset.

## Commencer

Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour les commandes Git.
Lire sa [fiche de mission](docs/missions/) et les [contrats communs](docs/CONTRATS.md) avant de coder.
Les missions peuvent préparer leurs fonctions sur des données fictives en attendant les livrables précédents.

Python 3.11 est la version de départ pour les contrôles du dépôt.
La mission 01 renseignera les dépendances ML après validation de leur compatibilité.

## Vérifications

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -p "test_*.py" -v
```

GitHub Actions lance ces commandes à chaque push et Pull Request vers main.
Aucun test métier n'est encore implémenté : une CI verte au démarrage vérifie seulement la syntaxe.
Chaque mission doit ajouter ses tests utiles ; voir [tests/README.md](tests/README.md).
L'entraînement complet et le téléchargement Kaggle ne sont pas exécutés dans cette CI.
