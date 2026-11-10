"""Mission 03 : Conversion vers NumPy.

À implémenter par le responsable de la mission.
Voir docs/missions/03-conversion-numpy.md pour le contrat et les critères.
Aucun téléchargement ou entraînement ne doit être lancé à l'import.
"""
"""Mission 03 : Conversion vers NumPy.

Transforme les ensembles PyTorch de la mission 02 en tableaux NumPy
utilisables par scikit-learn (missions 04 et 05).
Aucun téléchargement ou entraînement n'est lancé à l'import.
"""

import numpy as np
from torch.utils.data import DataLoader

TAILLE_IMAGE = 48
TAILLE_VECTEUR = TAILLE_IMAGE * TAILLE_IMAGE  # 2304 valeurs par image


def dataset_vers_numpy(dataset, taille_lot=256):
    """Convertit un ensemble PyTorch en tableaux NumPy (X, y).

    X : (n, 2304), float32, valeurs entre 0 et 1.
    y : (n,), indices de classe, dans le même ordre que X.
    """
    if len(dataset) == 0:
        raise ValueError("L'ensemble est vide : rien à convertir.")

    # shuffle=False : on garde l'ordre pour que image et label restent alignés.
    chargeur = DataLoader(dataset, batch_size=taille_lot, shuffle=False)

    lots_x = []
    lots_y = []
    for images, labels in chargeur:
        # (taille_lot, 1, 48, 48) -> (taille_lot, 2304)
        lots_x.append(images.reshape(images.shape[0], -1).numpy())
        lots_y.append(labels.numpy())

    X = np.concatenate(lots_x, axis=0)
    y = np.concatenate(lots_y, axis=0)
    return X, y
def verifier_numpy(X, y, nom="ensemble"):
    """Contrôle dimensions, types, valeurs finies et intervalle [0, 1].

    Lève une erreur explicite au premier problème trouvé.
    """
    if X.ndim != 2 or X.shape[1] != TAILLE_VECTEUR:
        raise ValueError(
            f"{nom} : X doit avoir la forme (n, {TAILLE_VECTEUR}), "
            f"reçu {X.shape}."
        )
    if X.shape[0] == 0:
        raise ValueError(f"{nom} : X est vide.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(
            f"{nom} : y doit avoir la forme ({X.shape[0]},), reçu {y.shape}."
        )
    if not np.issubdtype(X.dtype, np.floating):
        raise TypeError(f"{nom} : X doit être de type float, reçu {X.dtype}.")
    if not np.issubdtype(y.dtype, np.integer):
        raise TypeError(f"{nom} : y doit contenir des entiers, reçu {y.dtype}.")
    if not np.isfinite(X).all():
        raise ValueError(f"{nom} : X contient des valeurs NaN ou infinies.")
    if X.min() < 0.0 or X.max() > 1.0:
        raise ValueError(
            f"{nom} : les pixels doivent être dans [0, 1], "
            f"trouvé entre {X.min():.3f} et {X.max():.3f}."
        )


def convertir_donnees(train_ds, val_ds, test_ds, taille_lot=256):
    """Convertit train, validation et test en tableaux NumPy vérifiés.

    Renvoie X_train, y_train, X_val, y_val, X_test, y_test.
    """
    resultats = []
    for nom, ensemble in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        X, y = dataset_vers_numpy(ensemble, taille_lot)
        verifier_numpy(X, y, nom)
        resultats.extend([X, y])
    return tuple(resultats)
