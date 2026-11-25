# Mission 04 — Réduction PCA

Branche : `mission/04-pca`

Responsable : Lina.

Fichier : `src/emotion_detection/pca.py`

Dépend de : 03.

Livrable : pca, X_train_pca, X_val_pca, X_test_pca.

## Tâches et critères de réussite

- [x] Ajuster la PCA uniquement sur l'entraînement et transformer validation/test.
- [x] Étudier 100 à 200 composantes et tracer la variance expliquée cumulée.
- [x] Justifier le choix et conserver le même objet PCA pour toutes les transformations.
- [x] Coordonner avec la mission 05 : pour une validation croisée stricte, ajuster la PCA dans chaque pli via un Pipeline sklearn ; documenter tout écart au schéma pédagogique.
- [x] Documenter comment exécuter et vérifier la contribution.
- [ ] Faire relire puis intégrer la Pull Request dans main.

Lire [les contrats communs](../CONTRATS.md) et [le guide de contribution](../../CONTRIBUTING.md).

## Pourquoi une PCA ?

Chaque image de 48 x 48 pixels devient, après la mission 03, un vecteur de 2304 valeurs. Les pixels voisins se ressemblent beaucoup : une grande partie de ces 2304 valeurs est redondante. La PCA cherche les directions dans lesquelles les images varient le plus et ne garde que les premières. On obtient :

- des vecteurs beaucoup plus courts (150 valeurs au lieu de 2304), donc un entraînement plus rapide ;
- moins de bruit, car les dernières composantes codent surtout de petites variations sans intérêt ;
- moins de risque de sur-apprentissage pour la régression logistique.

La PCA est **non supervisée** : elle ne connaît pas les émotions. Elle garde ce qui varie le plus (éclairage, position du visage…), pas forcément ce qui distingue les émotions. C'est la régression logistique de la mission 05 qui exploite ensuite les composantes utiles.

## Ce que fait le script

Le fichier `pca.py` contient les fonctions suivantes :

| Fonction | Rôle |
| --- | --- |
| `reduire_donnees(X_train, X_val, X_test, n_composantes=150)` | **Livrable principal.** Ajuste la PCA sur le train avec `fit_transform`, puis projette la validation et le test avec `transform`. Renvoie `pca, X_train_pca, X_val_pca, X_test_pca`. |
| `etudier_variance(X_train)` | PCA complète sur le train ; renvoie aussi la variance expliquée cumulée. |
| `composantes_pour_variance(variance_cumulee, seuil)` | Nombre de composantes pour atteindre un seuil (par exemple 95 %). |
| `comparer_nombres_composantes(X_train, y_train, X_val, y_val)` | Pour 100, 125, 150, 175 et 200 composantes : variance gardée, accuracy train et validation d'une régression logistique simple (C = 1). Le test n'est pas utilisé. |
| `creer_pipeline_pca(n_composantes=150)` | `Pipeline(PCA, LogisticRegression)` proposé à la mission 05 (voir plus bas). |
| `tracer_variance`, `tracer_comparaison`, `afficher_composantes`, `afficher_reconstructions`, `afficher_projection_2d` | Figures de l'étude. Elles sont renvoyées à l'appelant, qui peut les afficher ou les enregistrer. |

Chaque fonction vérifie ses entrées et lève une erreur explicite si `X` n'a pas la forme `(n, 2304)`, contient des NaN, ou si le nombre de composantes est impossible.

La PCA est calculée de façon **exacte** (`svd_solver="covariance_eigh"`, rapide quand il y a beaucoup plus d'images que de pixels). Avec le réglage par défaut, scikit-learn utiliserait ici une méthode approchée et la variance affichée changerait légèrement d'une étape à l'autre. La graine 42 est fixée partout.

## Ce qu'on obtient

| Objet | Forme | Remarque |
| --- | --- | --- |
| `pca` | objet `PCA` ajusté | `pca.n_samples_` = taille du train : ajustée sur le train seulement |
| `X_train_pca` | `(20 096, 150)` | `fit_transform` sur le train |
| `X_val_pca` | `(4 306, 150)` | `transform` seulement |
| `X_test_pca` | `(4 307, 150)` | `transform` seulement |

On garde **le même objet `pca`** pour les trois ensembles : la validation et le test sont exprimés dans le repère appris sur le train, exactement comme une nouvelle image le serait après le déploiement.

## Choix du nombre de composantes

Le sujet demande de rester entre 100 et 200 composantes. Le choix retenu est **150 composantes**, au milieu de la plage. Il s'appuie sur trois éléments, tous calculés sans le jeu de test :

1. **La courbe de variance cumulée** (`outputs/pca_variance_cumulee.png`) : la variance gagnée par composante diminue très vite. Entre 150 et 200 composantes, on ne gagne que quelques points de variance.
2. **La comparaison sur la validation** (`outputs/pca_comparaison.png`) : l'accuracy de validation progresse peu au-delà de 150 composantes, alors que l'accuracy d'entraînement continue de monter. L'écart entre les deux grandit : les composantes en plus servent surtout à mémoriser le train.
3. **Les reconstructions** (`outputs/pca_reconstructions.png`) : avec 150 composantes, les traits du visage (yeux, sourcils, bouche) restent nettement visibles.

Résultats obtenus sur FER2013 avec `python -m src.emotion_detection.pca` (à remplir après l'exécution sur les 28 709 images) :

| Composantes | Variance gardée | Accuracy train | Accuracy validation |
| --- | --- | --- | --- |
| 100 | … % | … | … |
| 125 | … % | … | … |
| 150 | … % | … | … |
| 175 | … % | … | … |
| 200 | … % | … | … |

Seuils de variance affichés par le script : 90 % avec … composantes, 95 % avec … composantes.

Si les résultats de validation montraient un gain net au-delà de 150, il suffirait de lancer le script avec `--composantes 175` ou `--composantes 200` et de mettre à jour ce paragraphe.

## Accord avec la mission 05 : deux façons d'utiliser la PCA

**Schéma du sujet (par défaut).** La mission 05 reçoit `X_train_pca`, `X_val_pca`, `X_test_pca` et lance `GridSearchCV` dessus. C'est le schéma pédagogique des slides. Limite : la PCA a été ajustée sur tout le train, donc aussi sur les plis de validation internes du `GridSearchCV`. La fuite est faible (la PCA n'utilise pas les labels), mais elle existe.

**Schéma strict (proposé).** La mission 05 utilise `creer_pipeline_pca()` :

```python
from sklearn.model_selection import GridSearchCV
from src.emotion_detection.pca import creer_pipeline_pca

pipeline = creer_pipeline_pca(n_composantes=150)
clf = GridSearchCV(pipeline, {"logreg__C": [0.1, 1, 3, 10]}, cv=3, n_jobs=-1)
clf.fit(X_train, y_train)          # pixels bruts : la PCA est refaite dans chaque pli
clf.score(X_val, y_val)
```

Dans ce cas, `clf` reçoit **X brut** (2304 valeurs) et applique lui-même la PCA. **Il ne faut pas lui donner `X_*_pca`**, sinon la PCA serait appliquée deux fois. Le nom du paramètre devient `logreg__C`.

Le choix entre les deux schémas est à valider avec la mission 05 dans la PR commune, comme demandé dans [les contrats](../CONTRATS.md).

## Exécuter depuis la racine du dépôt

Les missions 02 et 03 doivent être présentes dans la branche, et le dossier `data/train` doit contenir les images (il n'est pas sur GitHub, voir la mission 02).

```bash
python -m pip install -r requirements-dev.txt
python -m src.emotion_detection.pca --data-dir data/train
```

Le terminal affiche :

- les formes d'entrée et de sortie ;
- le nombre de composantes pour 80, 90, 95 et 99 % de variance ;
- la variance gardée pour 100, 125, 150, 175 et 200 composantes ;
- le tableau de comparaison sur la validation.

Cinq figures sont enregistrées dans `outputs/` (ignoré par Git) : `pca_variance_cumulee.png`, `pca_composantes.png`, `pca_reconstructions.png`, `pca_comparaison.png` et `pca_projection_2d.png`.

Options utiles :

- `--composantes 200` : changer le nombre de composantes ;
- `--sans-comparaison` : ne pas lancer la comparaison, plus rapide ;
- `--dossier-figures autre/dossier` : enregistrer les figures ailleurs.

Durée indicative sur un processeur standard : moins d'une minute pour l'étude complète, sans carte graphique.

## Utiliser depuis une autre mission ou Colab

```python
from src.emotion_detection.preprocessing import preparer_donnees
from src.emotion_detection.conversion import convertir_donnees
from src.emotion_detection.pca import reduire_donnees

train_ds, val_ds, test_ds, class_to_idx = preparer_donnees("data/train")
X_train, y_train, X_val, y_val, X_test, y_test = convertir_donnees(
    train_ds, val_ds, test_ds
)
pca, X_train_pca, X_val_pca, X_test_pca = reduire_donnees(X_train, X_val, X_test)
```

Pour les figures dans Colab :

```python
import matplotlib.pyplot as plt
from src.emotion_detection.pca import (
    etudier_variance, tracer_variance, afficher_composantes, afficher_reconstructions,
)

pca_complete, variance_cumulee = etudier_variance(X_train)
tracer_variance(variance_cumulee)
afficher_composantes(pca_complete)
afficher_reconstructions(pca_complete, X_train)
plt.show()
```

## Vérifications rapides

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -p "test_*.py" -v
```

Les 11 tests de `tests/test_pca.py` utilisent de petites images fabriquées. Ils ne téléchargent pas FER2013 et n'entraînent aucun long modèle. Ils vérifient :

- les formes des sorties ;
- que la PCA est ajustée sur le train seulement : sa moyenne est celle du train, et la validation et le test sont une simple projection ;
- que changer la validation et le test ne modifie pas la PCA ;
- la reproductibilité ;
- que les composantes sont orthonormées ;
- le refus des entrées invalides ;
- la variance cumulée croissante jusqu'à 1 et le calcul des seuils ;
- la création des figures ;
- la comparaison des nombres de composantes ;
- le Pipeline, qui reçoit des pixels bruts et renvoie des probabilités valides.

Vérification locale : Python 3.13, scikit-learn 1.9 et numpy 2. Les 11 tests passent en moins d'une seconde. Le script complet a aussi été lancé sur des données simulées de la taille de FER2013 (20 096 / 4 306 / 4 307 images de 2304 valeurs) : il tourne en environ 20 secondes. L'exécution sur les vraies images reste à faire et à reporter dans le tableau ci-dessus. La CI du dépôt est configurée pour Python 3.11 avec `scikit-learn==1.6.1`, déclaré dans `requirements-dev.txt`.
