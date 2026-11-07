# emotion_detection_cv

Projet B3 : classification des expressions faciales sur FER2013 avec PyTorch pour le prétraitement, puis PCA et régression logistique scikit-learn.

## Mission 02 — Lancer le prétraitement PyTorch

Branche : **`mission/02-preprocessing`**.
Le code est dans [preprocessing.py](src/emotion_detection/preprocessing.py).

### Les images sont prêtes dans la copie locale

Les **28 709 images d'entraînement** ont été préparées à partir des lignes `Training` du fichier `fer2013.csv.zip`. Elles sont enregistrées en PNG 48 x 48, dans un dossier par émotion :

```text
data/train/
    angry/       3 995 images
    disgust/       436 images
    fear/        4 097 images
    happy/       7 215 images
    neutral/     4 965 images
    sad/         4 830 images
    surprise/    3 171 images
```

**Ces images sont présentes dans la copie locale utilisée pour cette mission.** Le dossier `data/` est ignoré par Git : les images ne sont pas envoyées sur GitHub et ne sont pas récupérées avec un clone ou un pull. Sur un autre ordinateur, il faut récupérer ce dossier d'images auprès du groupe ou utiliser la préparation de la mission 01 lorsqu'elle sera disponible.

Les lignes `PublicTest` et `PrivateTest` du CSV restent à part.

### Comment lancer le script

Ouvrir un terminal à la racine du dépôt `emotion_detection_cv`, sur la branche `mission/02-preprocessing`. Pour récupérer cette branche depuis GitHub la première fois :

```bash
git clone --branch mission/02-preprocessing https://github.com/mohsine92/emotion_detection_cv.git
cd emotion_detection_cv
```

Vérifier que le dossier `data/train` contenant les images est présent, puis installer les dépendances et lancer :

```bash
python -m pip install -r requirements-dev.txt
python -m src.emotion_detection.preprocessing --data-dir data/train
```

La commande doit être lancée depuis la racine du dépôt. Les versions CPU suffisent ; aucune carte graphique n'est nécessaire.

### Ce qu'on obtient

- Les images deviennent des tenseurs float `(1, 48, 48)`, avec des pixels entre 0 et 1.
- Le terminal affiche la correspondance émotion/indice et les effectifs : **20 096 pour l'entraînement, 4 306 pour la validation et 4 307 pour le test**.
- Une grille d'exemples par émotion s'affiche et est enregistrée dans `outputs/exemples_pretraitement.png`.
- Les jeux `train_ds`, `val_ds`, `test_ds` et `class_to_idx` sont prêts pour la mission 03.

Le découpage est reproductible avec la graine **42**. Il concerne seulement `data/train` : le `test_ds` créé ici est une partie de ce dossier, pas le test officiel de FER2013.

Pour récupérer les jeux dans un autre script ou dans Colab :

```python
from src.emotion_detection.preprocessing import preparer_donnees

train_ds, val_ds, test_ds, class_to_idx = preparer_donnees("data/train")
```

Le script a été exécuté sur les 28 709 images locales et les **6 tests passent**. Le [guide de la mission 02](docs/missions/02-preprocessing.md) détaille le découpage et les vérifications.

## Organisation

Nous travaillons d'abord dans GitHub avec des fichiers Python. Le notebook Google Colab sera assemblé plus tard pour le rendu.
Le prétraitement PyTorch de la mission 02 est disponible. Les autres modules restent à compléter.

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
Les tests de la mission 02 vérifient le prétraitement et le découpage sur de petites images locales.
Chaque mission ajoute ses tests utiles ; voir [tests/README.md](tests/README.md).
L'entraînement complet et le téléchargement Kaggle ne sont pas exécutés dans cette CI.
