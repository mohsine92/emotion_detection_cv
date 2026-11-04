# Travailler ensemble

## Première installation

```bash
git clone https://github.com/mohsine92/emotion_detection_cv.git
cd emotion_detection_cv
git fetch origin
git switch --track origin/mission/02-preprocessing
```

Remplacer la branche par celle de votre mission. Si elle existe déjà localement : `git switch mission/02-preprocessing`.

## Avant chaque session

```bash
git status
git fetch origin
git merge origin/main
```

Commencer avec un espace de travail propre : enregistrer ses modifications avant une fusion.
En cas de conflit, examiner les deux versions avec l'auteur concerné, corriger puis enregistrer la résolution. Ne pas écraser le travail des autres ni forcer un push.

## Enregistrer son travail

```bash
git add src/emotion_detection/preprocessing.py
git diff --cached
git commit -m "feat: ajouter le prétraitement des images"
git push
```

Ajouter aussi explicitement vos tests et documents. Vérifier le diff avant le commit.
Ne pas modifier les autres missions sans coordination.

## Intégrer

1. Ouvrir une Pull Request de sa branche vers `main`.
2. Indiquer l'issue avec `Closes #NUMERO`, expliquer les changements et les vérifications.
3. Obtenir la relecture d'un autre membre et vérifier la CI.
4. Fusionner quand la contribution est prête ; supprimer ensuite la branche terminée.
5. Les autres membres récupèrent `origin/main` dans leurs branches.

Ne pas attendre la fin du projet : intégrer de petits livrables cohérents.
Pour une correction ultérieure, créer une nouvelle branche depuis main à jour.

## Tests et dépendances

Utiliser `unittest` pour les tests rapides. Les fichiers se nomment `test_*.py`.
Si un test a besoin d'une bibliothèque, mettre à jour `requirements-dev.txt` et vérifier la CI.
La mission 01 documente les versions ML dans `requirements.txt`.
Aucun téléchargement de données ni entraînement long à l'import d'un module.
