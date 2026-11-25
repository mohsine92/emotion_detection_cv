"""Mission 04 : réduire les images avec une PCA (scikit-learn).

Depuis la racine du dépôt :
    python -m src.emotion_detection.pca --data-dir data/train

Entrées : X_train, X_val, X_test de la mission 03, de forme (n, 2304).
Sorties : pca, X_train_pca, X_val_pca, X_test_pca, de forme (n, nombre de composantes).
Aucun téléchargement ou entraînement n'est lancé à l'import.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

TAILLE_IMAGE = 48
TAILLE_VECTEUR = TAILLE_IMAGE * TAILLE_IMAGE  # 2304 valeurs par image
N_COMPOSANTES = 150  # choix retenu, justifié dans docs/missions/04-pca.md
PLAGE_ETUDIEE = (100, 125, 150, 175, 200)  # plage demandée par le sujet
GRAINE = 42
# Calcul exact et rapide quand il y a beaucoup plus d'images que de pixels,
# ce qui est le cas de FER2013 (scikit-learn >= 1.5). Sans ce réglage,
# scikit-learn utiliserait ici une méthode approchée et les chiffres
# varieraient légèrement d'une étape à l'autre.
SOLVEUR = "covariance_eigh"


def choisir_solveur(X):
    """Méthode exacte adaptée à la taille des données.

    Beaucoup d'images (FER2013) -> "covariance_eigh", rapide.
    Peu d'images (tests) -> "full", plus rapide dans ce cas.
    """
    return SOLVEUR if X.shape[0] >= X.shape[1] else "full"


def verifier_entree(X, nom="X"):
    """Contrôle qu'un tableau a la forme (n, 2304) et des valeurs finies."""
    if X.ndim != 2 or X.shape[1] != TAILLE_VECTEUR:
        raise ValueError(
            f"{nom} doit avoir la forme (n, {TAILLE_VECTEUR}), reçu {X.shape}."
        )
    if X.shape[0] == 0:
        raise ValueError(f"{nom} est vide.")
    if not np.isfinite(X).all():
        raise ValueError(f"{nom} contient des valeurs NaN ou infinies.")


def etudier_variance(X_train, graine=GRAINE):
    """Ajuste une PCA sur toutes les composantes du train.

    Sert uniquement à étudier la variance expliquée et à choisir le nombre
    de composantes. Renvoie la PCA complète et la variance expliquée cumulée.
    """
    verifier_entree(X_train, "X_train")
    pca_complete = PCA(svd_solver=choisir_solveur(X_train),
                       random_state=graine).fit(X_train)
    variance_cumulee = np.cumsum(pca_complete.explained_variance_ratio_)
    return pca_complete, variance_cumulee


def composantes_pour_variance(variance_cumulee, seuil):
    """Donne le plus petit nombre de composantes qui atteint le seuil.

    Exemple : seuil=0.95 -> nombre de composantes pour garder 95 % de variance.
    """
    if not 0 < seuil <= 1:
        raise ValueError(f"Le seuil doit être entre 0 et 1, reçu {seuil}.")
    # Petite tolérance : la somme des ratios peut valoir 0.9999999 au lieu de 1.
    atteint = variance_cumulee >= seuil - 1e-9
    if not atteint.any():
        raise ValueError(f"Le seuil {seuil} n'est jamais atteint.")
    return int(np.argmax(atteint) + 1)


def reduire_donnees(X_train, X_val, X_test, n_composantes=N_COMPOSANTES,
                    graine=GRAINE):
    """Ajuste la PCA sur le train uniquement, puis transforme les trois ensembles.

    Le même objet pca sert pour toutes les transformations : la validation et
    le test sont seulement projetés, jamais utilisés pour ajuster la PCA.
    Renvoie pca, X_train_pca, X_val_pca, X_test_pca.
    """
    for nom, X in (("X_train", X_train), ("X_val", X_val), ("X_test", X_test)):
        verifier_entree(X, nom)

    maximum = min(X_train.shape)
    if not 1 <= n_composantes <= maximum:
        raise ValueError(
            f"n_composantes doit être entre 1 et {maximum}, reçu {n_composantes}."
        )

    pca = PCA(n_components=n_composantes, svd_solver=choisir_solveur(X_train),
              random_state=graine)
    X_train_pca = pca.fit_transform(X_train)  # ajustement : train seulement
    X_val_pca = pca.transform(X_val)          # simple projection
    X_test_pca = pca.transform(X_test)        # simple projection
    return pca, X_train_pca, X_val_pca, X_test_pca


def comparer_nombres_composantes(X_train, y_train, X_val, y_val,
                                 nombres=PLAGE_ETUDIEE, C=1.0, graine=GRAINE):
    """Compare plusieurs nombres de composantes avec une régression logistique.

    Pour chaque nombre k, on garde les k premières composantes, on entraîne une
    régression logistique simple (C fixé) sur le train et on mesure l'accuracy
    sur la validation. Le test n'est jamais utilisé ici.

    Garder les k premières composantes d'une seule PCA revient à faire une PCA
    à k composantes : on n'ajuste donc qu'une PCA, avec le plus grand k.
    Renvoie une liste de dictionnaires, un par valeur de k.
    """
    nombres = sorted(nombres)
    verifier_entree(X_train, "X_train")
    verifier_entree(X_val, "X_val")

    pca = PCA(n_components=nombres[-1], svd_solver=choisir_solveur(X_train),
              random_state=graine).fit(X_train)
    Z_train = pca.transform(X_train)
    Z_val = pca.transform(X_val)
    variance_cumulee = np.cumsum(pca.explained_variance_ratio_)

    resultats = []
    for k in nombres:
        modele = LogisticRegression(C=C, solver="lbfgs", max_iter=2000,
                                    random_state=graine)
        modele.fit(Z_train[:, :k], y_train)
        resultats.append({
            "composantes": k,
            "variance": float(variance_cumulee[k - 1]),
            "accuracy_train": accuracy_score(y_train, modele.predict(Z_train[:, :k])),
            "accuracy_val": accuracy_score(y_val, modele.predict(Z_val[:, :k])),
        })
    return resultats


def creer_pipeline_pca(n_composantes=N_COMPOSANTES, modele=None, graine=GRAINE,
                       solveur=SOLVEUR):
    """Prépare un Pipeline(PCA, régression logistique) pour la mission 05.

    Dans un GridSearchCV, ce Pipeline réajuste la PCA dans chaque pli de la
    validation croisée : aucun pli de validation ne sert à ajuster la PCA.
    Attention : ce Pipeline reçoit X brut (2304 valeurs) ; il ne faut pas lui
    donner X_*_pca, sinon la PCA serait appliquée deux fois.
    Le solveur par défaut convient à FER2013 (beaucoup d'images).
    """
    if modele is None:
        modele = LogisticRegression(solver="lbfgs", max_iter=2000,
                                    random_state=graine)
    return Pipeline([
        ("pca", PCA(n_components=n_composantes, svd_solver=solveur,
                    random_state=graine)),
        ("logreg", modele),
    ])


def tracer_variance(variance_cumulee, n_choisi=N_COMPOSANTES,
                    plage=(PLAGE_ETUDIEE[0], PLAGE_ETUDIEE[-1])):
    """Trace la variance expliquée cumulée, avec la plage étudiée et le choix."""
    figure, axe = plt.subplots(figsize=(8, 4.5))
    nombres = np.arange(1, len(variance_cumulee) + 1)
    axe.plot(nombres, variance_cumulee * 100, color="#1f4e79")

    axe.axvspan(plage[0], plage[1], color="#f2c14e", alpha=0.25,
                label=f"plage étudiée ({plage[0]}–{plage[1]})")
    if n_choisi <= len(variance_cumulee):
        valeur = variance_cumulee[n_choisi - 1] * 100
        axe.axvline(n_choisi, color="#c0392b", linestyle="--",
                    label=f"choix : {n_choisi} composantes ({valeur:.1f} %)")

    for seuil in (0.90, 0.95):
        if variance_cumulee[-1] >= seuil:
            k = composantes_pour_variance(variance_cumulee, seuil)
            axe.axhline(seuil * 100, color="gray", linestyle=":", linewidth=1)
            axe.annotate(f"{int(seuil * 100)} % -> {k} comp.", (k, seuil * 100),
                         textcoords="offset points", xytext=(5, -12), fontsize=8)

    # On zoome sur les 500 premières composantes, où tout se joue.
    axe.set_xlim(0, min(500, len(variance_cumulee)))
    axe.set_ylim(0, 101)
    axe.set_xlabel("Nombre de composantes")
    axe.set_ylabel("Variance expliquée cumulée (%)")
    axe.set_title("PCA : variance expliquée cumulée (ajustée sur le train)")
    axe.legend(loc="lower right")
    figure.tight_layout()
    return figure


def tracer_comparaison(resultats):
    """Trace l'accuracy train et validation selon le nombre de composantes."""
    nombres = [r["composantes"] for r in resultats]
    figure, axe = plt.subplots(figsize=(7, 4))
    axe.plot(nombres, [r["accuracy_train"] for r in resultats], "o-", label="train")
    axe.plot(nombres, [r["accuracy_val"] for r in resultats], "o-", label="validation")
    axe.set_xlabel("Nombre de composantes")
    axe.set_ylabel("Accuracy")
    axe.set_title("Régression logistique (C fixé) selon le nombre de composantes")
    axe.legend()
    figure.tight_layout()
    return figure


def afficher_composantes(pca, nb_composantes=16):
    """Affiche les premières composantes principales comme des images 48 x 48.

    On les appelle souvent « eigenfaces » : rouge et bleu montrent les zones
    du visage qui varient ensemble, dans un sens ou dans l'autre.
    """
    nb_composantes = min(nb_composantes, pca.n_components_)
    colonnes = 8
    lignes = int(np.ceil(nb_composantes / colonnes))
    figure, axes = plt.subplots(lignes, colonnes, figsize=(colonnes * 1.8, lignes * 2.3),
                                squeeze=False)
    for i, axe in enumerate(axes.flat):
        axe.set_xticks([])
        axe.set_yticks([])
        if i >= nb_composantes:
            axe.set_visible(False)
            continue
        composante = pca.components_[i].reshape(TAILLE_IMAGE, TAILLE_IMAGE)
        limite = np.abs(composante).max()
        axe.imshow(composante, cmap="RdBu_r", vmin=-limite, vmax=limite)
        axe.set_title(f"PC{i + 1} ({pca.explained_variance_ratio_[i] * 100:.1f} %)",
                      fontsize=8)
    figure.suptitle("Premières composantes principales")
    figure.tight_layout()
    return figure


def afficher_reconstructions(pca_complete, X, nombres=(10, 50, 150, 300),
                             nb_images=4):
    """Montre des images reconstruites avec un nombre croissant de composantes.

    pca_complete doit avoir au moins max(nombres) composantes (par exemple
    celle renvoyée par etudier_variance). La dernière colonne est l'original.
    """
    nombres = [k for k in nombres if k <= pca_complete.n_components_]
    figure, axes = plt.subplots(nb_images, len(nombres) + 1,
                                figsize=((len(nombres) + 1) * 1.8, nb_images * 1.9),
                                squeeze=False)
    for ligne in range(nb_images):
        image = X[ligne]
        for colonne, k in enumerate(nombres):
            composantes = pca_complete.components_[:k]
            coordonnees = (image - pca_complete.mean_) @ composantes.T
            reconstruction = coordonnees @ composantes + pca_complete.mean_
            axes[ligne, colonne].imshow(
                reconstruction.reshape(TAILLE_IMAGE, TAILLE_IMAGE),
                cmap="gray", vmin=0, vmax=1,
            )
            if ligne == 0:
                axes[ligne, colonne].set_title(f"{k} comp.", fontsize=9)
        axes[ligne, -1].imshow(image.reshape(TAILLE_IMAGE, TAILLE_IMAGE),
                               cmap="gray", vmin=0, vmax=1)
        if ligne == 0:
            axes[ligne, -1].set_title("original", fontsize=9)
    for axe in axes.flat:
        axe.set_xticks([])
        axe.set_yticks([])
    figure.suptitle("Reconstruction des images selon le nombre de composantes")
    figure.tight_layout()
    return figure


def afficher_projection_2d(X_train_pca, y_train, class_to_idx, nb_points=4000):
    """Place les images du train sur les deux premières composantes."""
    figure, axe = plt.subplots(figsize=(7, 5.5))
    X_points, y_points = X_train_pca[:nb_points], y_train[:nb_points]
    for nom, indice in class_to_idx.items():
        masque = y_points == indice
        axe.scatter(X_points[masque, 0], X_points[masque, 1], s=5, alpha=0.5,
                    label=nom)
    axe.set_xlabel("PC1")
    axe.set_ylabel("PC2")
    axe.set_title("Images du train projetées sur PC1 et PC2")
    axe.legend(markerscale=3, fontsize=8)
    figure.tight_layout()
    return figure


def main():
    """Lance l'étude PCA complète depuis le terminal et enregistre les figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train",
                        help="Dossier contenant les 7 émotions.")
    parser.add_argument("--composantes", type=int, default=N_COMPOSANTES,
                        help="Nombre de composantes gardées.")
    parser.add_argument("--sans-comparaison", action="store_true",
                        help="Ne pas lancer la comparaison 100–200 (plus rapide).")
    parser.add_argument("--dossier-figures", default="outputs",
                        help="Dossier où enregistrer les figures.")
    args = parser.parse_args()

    # Imports ici : ils ne sont utiles qu'avec les vraies données.
    from src.emotion_detection.conversion import convertir_donnees
    from src.emotion_detection.preprocessing import preparer_donnees

    train_ds, val_ds, test_ds, class_to_idx = preparer_donnees(args.data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = convertir_donnees(
        train_ds, val_ds, test_ds
    )
    print(f"Entrées : X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    dossier = Path(args.dossier_figures)
    dossier.mkdir(parents=True, exist_ok=True)

    # 1. Étude de la variance expliquée
    pca_complete, variance_cumulee = etudier_variance(X_train)
    for seuil in (0.80, 0.90, 0.95, 0.99):
        k = composantes_pour_variance(variance_cumulee, seuil)
        print(f"{int(seuil * 100)} % de variance : {k} composantes")
    for k in PLAGE_ETUDIEE:
        print(f"{k} composantes : {variance_cumulee[k - 1] * 100:.1f} % de variance")

    figures = {
        "pca_variance_cumulee.png": tracer_variance(variance_cumulee, args.composantes),
        "pca_composantes.png": afficher_composantes(pca_complete),
        "pca_reconstructions.png": afficher_reconstructions(pca_complete, X_train),
    }

    # 2. Comparaison 100–200 composantes sur la validation
    if not args.sans_comparaison:
        resultats = comparer_nombres_composantes(X_train, y_train, X_val, y_val)
        print("\ncomposantes | variance | accuracy train | accuracy validation")
        for r in resultats:
            print(f"{r['composantes']:>11} | {r['variance'] * 100:7.1f} % | "
                  f"{r['accuracy_train']:14.4f} | {r['accuracy_val']:.4f}")
        figures["pca_comparaison.png"] = tracer_comparaison(resultats)

    # 3. Réduction finale, transmise à la mission 05
    pca, X_train_pca, X_val_pca, X_test_pca = reduire_donnees(
        X_train, X_val, X_test, args.composantes
    )
    print(f"\nSorties : X_train_pca={X_train_pca.shape}, X_val_pca={X_val_pca.shape}, "
          f"X_test_pca={X_test_pca.shape}")
    print(f"Variance conservée : {pca.explained_variance_ratio_.sum() * 100:.1f} %")
    figures["pca_projection_2d.png"] = afficher_projection_2d(
        X_train_pca, y_train, class_to_idx
    )

    for nom, figure in figures.items():
        figure.savefig(dossier / nom, dpi=150)
        print(f"Figure enregistrée : {dossier / nom}")
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
