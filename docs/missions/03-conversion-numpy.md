# Mission 03 — Conversion vers NumPy

Branche : `mission/03-conversion-numpy`

Responsable : Nesrine Latrache.

Fichier : `src/emotion_detection/conversion.py`

Dépend de : 02.

Livrable : X_train, y_train, X_val, y_val, X_test, y_test.

## Tâches et critères de réussite

- [x] Construire les DataLoaders et une fonction de conversion réutilisable.
- [x] Aplatir chaque image en 2304 valeurs et préserver l'association image/label.
- [x] Vérifier dimensions, types, valeurs finies et intervalle [0, 1].
- [x] Ajouter des tests sur de petits lots, sans FER2013.
- [x] Documenter comment exécuter et vérifier la contribution.


Lire [les contrats communs](../CONTRATS.md) et [le guide de contribution](../../CONTRIBUTING.md).

## Ce que fait le script

Le fichier `conversion.py` contient trois fonctions :

- `dataset_vers_numpy(dataset, taille_lot=256)` : parcourt un ensemble PyTorch avec un `DataLoader`, aplatit chaque image `(1, 48, 48)` en 2304 valeurs et renvoie `(X, y)`.
- `verifier_numpy(X, y, nom="ensemble")` : contrôle les dimensions, les types, l'absence de NaN ou d'infini et l'intervalle [0, 1]. Elle lève une erreur explicite au premier problème.
- `convertir_donnees(train_ds, val_ds, test_ds, taille_lot=256)` : convertit et vérifie les trois ensembles, puis renvoie `X_train, y_train, X_val, y_val, X_test, y_test`.

Le `DataLoader` est créé avec `shuffle=False` : l'ordre est conservé, donc la ligne `i` de `X` correspond toujours à `y[i]`. Le résultat ne dépend pas de `taille_lot`. L'aplatissement se fait ligne par ligne : le pixel `(ligne, colonne)` se retrouve à la position `ligne * 48 + colonne`.

## Ce qu'on obtient

| Objet | Forme | Type |
| --- | --- | --- |
| `X_train`, `X_val`, `X_test` | `(n, 2304)` | `float32`, valeurs entre 0 et 1 |
| `y_train`, `y_val`, `y_test` | `(n,)` | entiers (indices de classe) |

Les pixels sont déjà entre 0 et 1 grâce à `ToTensor()` de la mission 02? on ne les divise pas une seconde fois par 255. Les indices de classe suivent le `class_to_idx` de la mission 02.

## Utiliser depuis une autre mission ou Colab

```python
from src.emotion_detection.preprocessing import preparer_donnees
from src.emotion_detection.conversion import convertir_donnees

train_ds, val_ds, test_ds, class_to_idx = preparer_donnees("data/train")
X_train, y_train, X_val, y_val, X_test, y_test = convertir_donnees(
    train_ds, val_ds, test_ds
)
```

Ces tableaux sont prêts pour la PCA et la régression logistique (missions 04 et 05).

## Exécuter depuis la racine du dépôt

Le dossier `data/train` doit être présent (il n'est pas sur GitHub, voir la mission 02). Avec les 28 709 images, les effectifs attendus sont 20 096 / 4 306 / 4 307 :

```bash
python -c "from src.emotion_detection.preprocessing import preparer_donnees; from src.emotion_detection.conversion import convertir_donnees; a, b, c, _ = preparer_donnees('data/train'); r = convertir_donnees(a, b, c); print([x.shape for x in r])"
```

## Vérifications rapides

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -p "test_*.py" -v
```

Les tests de `tests/test_conversion.py` utilisent de petits lots fabriqués : formes et types, association image/label, aplatissement ligne par ligne, indépendance vis-à-vis de la taille de lot, ensemble vide refusé, valeurs et formes invalides refusées, et passage complet avec la sortie de la mission 02. Ils ne téléchargent pas FER2013 et n'entraînent aucun modèle.

Vérification locale effectuée sous Windows avec Python 3.14, torch 2.15.0+cpu et numpy 2.5.4. Les 16 tests passent (6 de la mission 02 et 10 de la mission 03). La CI du dépôt reste configurée pour Python 3.11 elle n'a pas été vérifiée à distance.
