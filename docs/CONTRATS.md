# Contrats communs

- Chemins relatifs ; données locales dans ./data.
- Graine commune : 42 ; fixer aussi les générateurs utilisés pour les splits et mélanges.
- Conserver une seule correspondance class_to_idx issue des données et la transmettre. Ne pas inventer un ordre alphabétique français.
- PyTorch : images float de forme (1, 48, 48), valeurs [0, 1].
- train_ds, val_ds, test_ds : ensembles disjoints, protocole de découpage documenté par la mission 02.
- X_train, X_val, X_test : tableaux NumPy (n, 2304), valeurs finies.
- y_train, y_val, y_test : vecteurs (n,), indices de classe alignés avec X.
- pca : ajustée sur l'entraînement uniquement ; X_*_pca : (n, nombre de composantes).
- clf : modèle final sélectionné sans utiliser le jeu de test.
- y_pred : une prédiction par exemple de test ; métriques avec les vrais noms de classes.

## Accord PCA / modèle

Le guide pédagogique transmet X_*_pca au GridSearchCV. Pour éviter aussi la fuite entre les plis internes de validation croisée, les missions 04 et 05 peuvent utiliser un Pipeline(PCA, LogisticRegression) dans GridSearchCV.
Dans ce cas, clf reçoit X brut et applique lui-même la PCA ; l'évaluation ne doit pas transformer une seconde fois.
Documenter le choix dans la PR commune avant l'intégration.

## Dépendances de travail

01 -> 02 -> 03 -> 04 -> 05 -> 06.
Préparer les modules et tests avec de petites données fictives permet de démarrer en parallèle.
Toute modification d'une entrée/sortie doit être discutée avec la mission suivante.
