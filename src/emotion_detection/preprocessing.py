"""Mission 02 : préparer les images FER2013 avec PyTorch.

Depuis la racine du dépôt :
    python -m src.emotion_detection.preprocessing --data-dir data/train

Les images doivent déjà être rangées dans un dossier par émotion.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms


def creer_transformations():
    """Donne le même format à toutes les images : un canal, 48 x 48 pixels."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        # ToTensor convertit aussi les pixels de [0, 255] vers [0, 1].
        transforms.ToTensor(),
    ])


def preparer_donnees(dossier_images="data/train", graine=42):
    """Charge un dossier d'images et le découpe en 70 % / 15 % / 15 %.

    Renvoie train_ds, val_ds, test_ds et la correspondance class_to_idx.
    Seul le dossier indiqué est utilisé : data/test reste à part.
    """
    dossier_images = Path(dossier_images)
    if not dossier_images.is_dir():
        raise FileNotFoundError(
            f"Dossier d'images introuvable : {dossier_images}. "
            "La mission 01 doit fournir les images dans data/train, "
            "avec un sous-dossier par émotion. Le CSV brut ne suffit pas."
        )

    # Évite de prendre train et test pour deux classes d'émotions.
    if (dossier_images / "train").is_dir() or (dossier_images / "test").is_dir():
        raise ValueError(
            "Indique le dossier des émotions, par exemple data/train, "
            "plutôt que le dossier data qui contient les partitions."
        )

    dataset = datasets.ImageFolder(
        root=dossier_images,
        transform=creer_transformations(),
    )
    if len(dataset.classes) != 7:
        raise ValueError(
            f"FER2013 doit contenir 7 classes ; trouvé : {dataset.classes}."
        )

    # On calcule le dernier effectif par différence pour garder toutes les images.
    nombre_images = len(dataset)
    taille_train = int(0.70 * nombre_images)
    taille_val = int(0.15 * nombre_images)
    taille_test = nombre_images - taille_train - taille_val

    # Le même dossier et la même graine redonnent le même découpage.
    generateur = torch.Generator().manual_seed(graine)
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [taille_train, taille_val, taille_test],
        generator=generateur,
    )

    # On garde les indices attribués par ImageFolder pour la suite du projet.
    class_to_idx = dataset.class_to_idx.copy()
    return train_ds, val_ds, test_ds, class_to_idx


def afficher_exemples(dataset, nb_exemples=3):
    """Prépare une grille d'images transformées, avec une ligne par émotion.

    dataset est l'ImageFolder partagé par les trois sous-ensembles.
    La figure renvoyée peut être affichée ou enregistrée par l'appelant.
    """
    if nb_exemples < 1:
        raise ValueError("Il faut afficher au moins un exemple par classe.")

    figure, axes = plt.subplots(
        len(dataset.classes),
        nb_exemples,
        figsize=(nb_exemples * 2, len(dataset.classes) * 1.7),
        squeeze=False,
    )

    for nom_classe, indice_classe in dataset.class_to_idx.items():
        # On prend les premiers fichiers de chaque classe pour une figure stable.
        indices = [
            i for i, label in enumerate(dataset.targets)
            if label == indice_classe
        ][:nb_exemples]

        for colonne in range(nb_exemples):
            axe = axes[indice_classe, colonne]
            axe.set_xticks([])
            axe.set_yticks([])
            if colonne == 0:
                axe.set_ylabel(nom_classe)

            if colonne < len(indices):
                image, _ = dataset[indices[colonne]]
                # On retire seulement le canal pour l'affichage en niveaux de gris.
                axe.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                axe.set_visible(False)

    figure.suptitle("FER2013 : images prétraitées en 48 x 48")
    figure.tight_layout()
    return figure


def main():
    """Lance une vérification visuelle depuis le terminal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default="data/train", help="Dossier contenant les 7 émotions."
    )
    parser.add_argument(
        "--figure", default="outputs/exemples_pretraitement.png",
        help="Emplacement de la grille d'exemples."
    )
    args = parser.parse_args()

    train_ds, val_ds, test_ds, class_to_idx = preparer_donnees(args.data_dir)
    print("Correspondance émotion -> indice :", class_to_idx)
    print(f"Images : train={len(train_ds)}, validation={len(val_ds)}, test={len(test_ds)}")

    image, label = train_ds[0]
    print(f"Exemple : forme={tuple(image.shape)}, type={image.dtype}, label={label}")
    print(f"Pixels entre {image.min().item():.3f} et {image.max().item():.3f}")

    figure = afficher_exemples(train_ds.dataset)
    chemin_figure = Path(args.figure)
    chemin_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chemin_figure, dpi=150)
    print(f"Grille enregistrée : {chemin_figure}")
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
