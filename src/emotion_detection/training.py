"""Mission 05 : régression logistique multinomiale + GridSearchCV.

Ce module reprend le code de la Section 5 du notebook :
- entrainer_logreg      : GridSearchCV sur C (modèle officiel `clf`)
- tableau_scores        : scores train / validation croisée pour chaque C
- tracer_scores_c       : courbe accuracy en fonction de C
- comparer_modeles      : bonus, comparaison avec Dummy, KNN et LogReg équilibrée
- rappel_par_emotion    : bonus, rappel de chaque émotion sur la validation

Les fonctions ne font jamais appel au jeu de test (réservé à la Section 6).
Les graphiques sont tracés sans plt.show() : c'est le notebook qui l'appelle.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

GRILLE_C = [0.001, 0.01, 0.1, 1, 3, 10, 100]
GRILLE_K = [5, 11, 21, 41]
MAX_ITER = 1000
N_PLIS = 3


def entrainer_logreg(X_train_pca, y_train, grille_c=GRILLE_C, class_weight=None):
    """Entraîne une régression logistique et choisit C par validation croisée.

    Avec solver="lbfgs", le modèle multinomial (softmax) et la pénalité L2 sont
    les comportements par défaut : on ne passe donc pas `multi_class`
    (paramètre déprécié dans les versions récentes de scikit-learn).

    Retourne le GridSearchCV déjà réajusté (refit) sur tout le train.
    """
    logreg = LogisticRegression(
        solver="lbfgs", max_iter=MAX_ITER, class_weight=class_weight
    )
    clf = GridSearchCV(
        logreg,
        {"C": grille_c},
        cv=N_PLIS,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True,
    )
    clf.fit(X_train_pca, y_train)
    return clf


def afficher_resultat(clf, X_val_pca, y_val):
    """Affiche le meilleur C et les accuracies (validation croisée et validation)."""
    print("Meilleur C :", clf.best_params_["C"])
    print(f"Accuracy CV (train, {N_PLIS} plis) : {clf.best_score_:.3f}")
    print(f"Accuracy validation         : {clf.score(X_val_pca, y_val):.3f}")


def tableau_scores(clf):
    """Tableau des scores obtenus pour chaque valeur de C."""
    res = pd.DataFrame(clf.cv_results_)[
        ["param_C", "mean_train_score", "mean_test_score", "std_test_score"]
    ].copy()
    res.columns = ["C", "accuracy train", "accuracy CV", "écart-type CV"]
    res["C"] = res["C"].astype(float)
    return res


def tracer_scores_c(tableau):
    """Courbe de l'accuracy (train et validation croisée) en fonction de C."""
    plt.figure(figsize=(7, 4))
    plt.semilogx(tableau["C"], tableau["accuracy train"], "o-", label="train")
    plt.semilogx(
        tableau["C"], tableau["accuracy CV"], "o-", label="validation croisée"
    )
    plt.xlabel("C (échelle log)")
    plt.ylabel("Accuracy")
    plt.title("Effet de la régularisation sur l'accuracy")
    plt.legend()
    plt.grid(alpha=0.3)


def comparer_modeles(clf, X_train_pca, y_train, X_val_pca, y_val):
    """Bonus : compare `clf` à un modèle naïf, un KNN et une LogReg équilibrée.

    La comparaison se fait sur la validation. Retourne (tableau, modeles),
    où `modeles` est un dictionnaire nom -> modèle entraîné.
    """
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train_pca, y_train)

    knn = GridSearchCV(
        KNeighborsClassifier(), {"n_neighbors": GRILLE_K}, cv=N_PLIS, n_jobs=-1
    ).fit(X_train_pca, y_train)

    logreg_bal = entrainer_logreg(X_train_pca, y_train, class_weight="balanced")

    modeles = {
        "Dummy (classe majoritaire)": dummy,
        "LogReg (clf)": clf,
        "LogReg class_weight=balanced": logreg_bal,
        "KNN": knn,
    }
    tableau = pd.DataFrame(
        {
            "Modèle": list(modeles),
            "Accuracy validation": [
                m.score(X_val_pca, y_val) for m in modeles.values()
            ],
        }
    )
    return tableau, modeles


def rappel_par_emotion(modeles, X_val_pca, y_val, class_to_idx):
    """Bonus : rappel de chaque émotion sur la validation (LogReg vs équilibrée)."""
    noms = sorted(class_to_idx, key=class_to_idx.get)
    return pd.DataFrame(
        {
            "LogReg": recall_score(
                y_val, modeles["LogReg (clf)"].predict(X_val_pca), average=None
            ),
            "LogReg balanced": recall_score(
                y_val,
                modeles["LogReg class_weight=balanced"].predict(X_val_pca),
                average=None,
            ),
        },
        index=noms,
    )