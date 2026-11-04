# Répartition du groupe

| Mission | Responsable GitHub | Relecteur | Date cible |
| --- | --- | --- | --- |
| 01 — Environnement et données | À attribuer | À attribuer | À définir |
| 02 — Prétraitement PyTorch | À attribuer | À attribuer | À définir |
| 03 — Conversion vers NumPy | À attribuer | À attribuer | À définir |
| 04 — Réduction PCA | À attribuer | À attribuer | À définir |
| 05 — Régression logistique et GridSearchCV | À attribuer | À attribuer | À définir |
| 06 — Évaluation et conclusion | À attribuer | À attribuer | À définir |

Utiliser les issues GitHub comme source de vérité pour l'avancement : ouverte = à faire/en cours ; fermée par une PR = intégrée.
Choisir un responsable par mission via Assignees et annoncer le début du travail dans l'issue.
Choisir aussi un référent d'intégration pour vérifier le pipeline complet.
Ne pas fermer une issue uniquement parce qu'une branche existe.

## Réglages du dépôt

Règle d'équipe : pas de push direct sur main après la mise en place initiale.
Le propriétaire peut la rendre obligatoire dans Settings > Rules > Rulesets (selon les possibilités du compte) :
- cibler main ;
- exiger une Pull Request et une approbation ;
- exiger le contrôle "Vérifications Python", après son premier passage ;
- bloquer les force-pushs et la suppression de main.

Ces réglages ne sont pas activés par les fichiers du dépôt. Ajouter les collaborateurs dans Settings > Collaborators.
