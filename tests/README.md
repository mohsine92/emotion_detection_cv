# Tests rapides

Les tests de la mission 02 sont dans `test_preprocessing.py`. Ils utilisent des images temporaires et ne nécessitent pas FER2013. Les autres missions ajouteront leurs tests avec leurs fonctions.

- Mission 02 : image synthétique -> tenseur (1, 48, 48), valeurs [0, 1] ; séparation sans chevauchement et reproductible.
- Mission 03 : petits lots -> X de forme (n, 2304), y de forme (n,), association image/label conservée.
- Mission 04 : données synthétiques -> formes réduites cohérentes ; validation/test ne réajustent pas la PCA. Un petit nombre de composantes suffit pour ce test.
- Mission 05 : petit jeu synthétique multiclasses -> fit/predict fonctionne, probabilités valides. Ne pas imposer un seuil arbitraire d'accuracy sur FER2013.
- Mission 06 : labels connus -> métriques et matrice de confusion cohérentes.
- Intégration : petit pipeline synthétique, sans Kaggle ni entraînement complet.

Commande depuis la racine : `python -m unittest discover -s tests -p "test_*.py" -v`.
Imports : `from src.emotion_detection import conversion`, par exemple.
Déclarer les dépendances des tests dans requirements-dev.txt.
