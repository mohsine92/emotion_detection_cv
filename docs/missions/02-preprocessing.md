# Mission 02 — Prétraitement PyTorch

Branche : `mission/02-preprocessing`

Responsable : à attribuer dans l'issue GitHub.

Fichier : `src/emotion_detection/preprocessing.py`

Dépend de : 01.

Livrable : train_ds, val_ds, test_ds, class_to_idx.

## Tâches et critères de réussite

- [x] Appliquer Grayscale, Resize((48, 48)) et ToTensor.
- [x] Documenter le périmètre et le découpage 70/15/15 proposé dans le sujet ; ne pas mélanger silencieusement les splits existants.
- [x] Fixer la graine et vérifier que les sous-ensembles ne se chevauchent pas.
- [x] Afficher des exemples et transmettre une correspondance des classes commune.
- [x] Documenter comment exécuter et vérifier la contribution.
- [ ] Faire relire puis intégrer la Pull Request dans main.

Lire [les contrats communs](../CONTRATS.md) et [le guide de contribution](../../CONTRIBUTING.md).

## Ce que fait le script

Le fichier `preprocessing.py` contient trois fonctions principales :

- `creer_transformations()` : niveaux de gris, taille 48 x 48 et conversion en tenseur.
- `preparer_donnees(dossier_images="data/train", graine=42)` : chargement avec ImageFolder et découpage des images.
- `afficher_exemples(dataset, nb_exemples=3)` : création d'une grille, avec une ligne par émotion.

Chaque image devient un tenseur float de forme `(1, 48, 48)`. `ToTensor()` place déjà les pixels entre 0 et 1 : on ne les divise pas une seconde fois par 255.
Le script garde les images sous cette forme. La conversion en vecteurs NumPy appartient à la mission 03.

## Données attendues

La mission 01 doit préparer un dossier contenant directement les sept émotions :

```text
data/train/
    angry/       image_001.png ...
    disgust/     image_002.png ...
    fear/        image_003.png ...
    happy/       image_004.png ...
    neutral/     image_005.png ...
    sad/         image_006.png ...
    surprise/    image_007.png ...
```

Le fichier `fer2013.csv.zip` fourni contient des pixels dans un CSV, pas ces dossiers. S'il est utilisé, une préparation locale doit d'abord reconstruire les images 48 x 48 en conservant la colonne `Usage`. Seules les lignes `Training` alimentent `data/train` ; `PublicTest` et `PrivateTest` restent distincts. Cette préparation ne se déclenche pas à l'import du module.

On utilise toujours le `class_to_idx` lu par ImageFolder. Son ordre peut différer des numéros d'émotions du CSV : par exemple, avec ces noms de dossiers, `neutral` vaut 4 alors que son numéro CSV vaut 6.

## Choix du découpage

On suit l'exemple du sujet : **seul `data/train` est redécoupé en 70 / 15 / 15**. Le dossier `data/test` existant n'est ni chargé ni fusionné avec lui. Le `test_ds` renvoyé est donc une partie de `data/train`, pas le test officiel de FER2013.

Les effectifs train et validation sont arrondis vers le bas ; le test reçoit le reste. Pour 28 709 images, cela donne **20 096 / 4 306 / 4 307** images. `random_split` utilise son propre générateur avec la graine 42. Avec les mêmes fichiers et la même graine, les indices sont identiques à chaque exécution.

Les trois ensembles ne partagent aucun indice de fichier. Cela ne supprime pas les doublons de contenu déjà présents dans FER2013. Le découpage est aléatoire, sans stratification par émotion ; ces limites devront être prises en compte lors de l'évaluation.

## Exécuter depuis la racine du dépôt

Installer les dépendances de cette mission (CPU, sans carte graphique) :

```bash
python -m pip install -r requirements-dev.txt
```

Lancer le script une fois le dossier d'images préparé :

```bash
python -m src.emotion_detection.preprocessing --data-dir data/train
```

Le terminal affiche les classes, les effectifs et les caractéristiques d'une image. Une grille s'affiche et est enregistrée dans `outputs/exemples_pretraitement.png` (ignoré par Git). L'option `--figure` permet de choisir un autre emplacement.

## Transmettre les données à la mission 03 ou au notebook Colab

```python
from src.emotion_detection.preprocessing import preparer_donnees

train_ds, val_ds, test_ds, class_to_idx = preparer_donnees("data/train")
```

Pour afficher la grille dans Colab :

```python
import matplotlib.pyplot as plt
from src.emotion_detection.preprocessing import afficher_exemples

figure = afficher_exemples(train_ds.dataset)
plt.show()
```

## Vérifications rapides

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -p "test_*.py" -v
```

Les tests créent de petites images temporaires : format `(1, 48, 48)`, bornes 0 et 1, association image/label, découpage complet et sans chevauchement, reproductibilité, grille et erreurs de dossier. Ils ne téléchargent pas FER2013 et n'entraînent aucun modèle.

Vérification locale effectuée sous Windows avec Python 3.10.11, torch 2.11.0+cpu et torchvision 0.26.0+cpu : les six tests passent et la compilation réussit. Le script a aussi été exécuté sur les 28 709 images `Training` du CSV fourni : effectifs 20 096 / 4 306 / 4 307 et grille créée. Les images préparées localement restent dans `data/`, ignoré par Git. La CI du dépôt reste configurée pour Python 3.11 ; elle n'a pas été exécutée à distance.
