"""Vérifications rapides du prétraitement, sans télécharger FER2013."""

import tempfile
import unittest
from pathlib import Path

import matplotlib

# Les tests créent les figures en mémoire, sans ouvrir de fenêtre.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.emotion_detection.preprocessing import (
    afficher_exemples,
    creer_transformations,
    preparer_donnees,
)


class TestPretraitement(unittest.TestCase):
    """On utilise de petites images créées pour chaque test."""

    def setUp(self):
        self.dossier_temporaire = tempfile.TemporaryDirectory()
        self.addCleanup(self.dossier_temporaire.cleanup)
        self.addCleanup(plt.close, "all")

        self.racine = Path(self.dossier_temporaire.name)
        self.dossier_train = self.racine / "data" / "train"
        self.classes = [
            "surprise", "happy", "angry", "fear", "sad", "neutral", "disgust"
        ]
        self.correspondance_attendue = {
            nom: indice for indice, nom in enumerate(sorted(self.classes))
        }

        # Chaque émotion a une intensité différente pour vérifier ses labels.
        for nom, indice in self.correspondance_attendue.items():
            dossier_classe = self.dossier_train / nom
            dossier_classe.mkdir(parents=True)
            intensite = 30 + indice * 20
            for numero in range(4):
                image = Image.new("RGB", (29, 61), (intensite,) * 3)
                image.save(dossier_classe / f"image_{numero}.png")

        # Ce dossier existe déjà dans FER2013 et doit rester à part.
        dossier_test = self.racine / "data" / "test" / "angry"
        dossier_test.mkdir(parents=True)
        Image.new("L", (48, 48), 255).save(dossier_test / "image_test.png")

    def test_image_rgb_convertie_en_tenseur_gris_48_par_48(self):
        image = Image.new("RGB", (80, 25), (200, 100, 50))

        tenseur = creer_transformations()(image)

        self.assertEqual(tuple(tenseur.shape), (1, 48, 48))
        self.assertEqual(tenseur.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tenseur).all().item())
        self.assertGreaterEqual(tenseur.min().item(), 0.0)
        self.assertLessEqual(tenseur.max().item(), 1.0)

    def test_noir_et_blanc_donnent_zero_et_un(self):
        transformation = creer_transformations()

        # Le blanc à 1 permet aussi de repérer une double division par 255.
        for intensite, valeur_attendue in [(0, 0.0), (255, 1.0)]:
            with self.subTest(intensite=intensite):
                image = Image.new("RGB", (20, 70), (intensite,) * 3)
                tenseur = transformation(image)
                self.assertTrue(torch.all(tenseur == valeur_attendue).item())

    def test_decoupage_complet_sans_indices_en_commun(self):
        train_ds, val_ds, test_ds, _ = preparer_donnees(self.dossier_train)

        self.assertEqual([len(train_ds), len(val_ds), len(test_ds)], [19, 4, 5])
        self.assertIs(train_ds.dataset, val_ds.dataset)
        self.assertIs(train_ds.dataset, test_ds.dataset)

        indices_train = set(train_ds.indices)
        indices_val = set(val_ds.indices)
        indices_test = set(test_ds.indices)
        self.assertTrue(indices_train.isdisjoint(indices_val))
        self.assertTrue(indices_train.isdisjoint(indices_test))
        self.assertTrue(indices_val.isdisjoint(indices_test))
        self.assertEqual(indices_train | indices_val | indices_test, set(range(28)))

        # Le fichier du dossier data/test n'entre pas dans ce découpage.
        for chemin, _ in train_ds.dataset.samples:
            self.assertTrue(Path(chemin).is_relative_to(self.dossier_train))

    def test_decoupage_reproductible_malgre_le_hasard_global(self):
        etat_initial = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, etat_initial)

        torch.manual_seed(123)
        premiers = preparer_donnees(self.dossier_train, graine=42)
        torch.manual_seed(987)
        torch.rand(100)
        suivants = preparer_donnees(self.dossier_train, graine=42)

        for premier, suivant in zip(premiers[:3], suivants[:3]):
            self.assertEqual(list(premier.indices), list(suivant.indices))

    def test_classes_labels_et_exemples_restent_coherents(self):
        train_ds, val_ds, test_ds, class_to_idx = preparer_donnees(
            self.dossier_train
        )
        self.assertEqual(class_to_idx, self.correspondance_attendue)

        # La couleur connue de chaque classe doit rester liée au bon label.
        for sous_ensemble in (train_ds, val_ds, test_ds):
            for image, label in sous_ensemble:
                self.assertAlmostEqual(image[0, 0, 0].item(), (30 + label * 20) / 255)

        figure = afficher_exemples(train_ds.dataset, nb_exemples=3)
        self.assertEqual(len(figure.axes), 7 * 3)
        self.assertTrue(all(len(axe.images) == 1 for axe in figure.axes))

    def test_erreurs_explicites_pour_un_mauvais_dossier(self):
        with self.assertRaises(FileNotFoundError):
            preparer_donnees(self.racine / "dossier_absent")

        # Charger data au lieu de data/train créerait les classes train et test.
        with self.assertRaisesRegex(ValueError, "data/train"):
            preparer_donnees(self.racine / "data")

        dossier_incomplet = self.racine / "deux_classes"
        for nom in ("angry", "happy"):
            dossier_classe = dossier_incomplet / nom
            dossier_classe.mkdir(parents=True)
            Image.new("L", (48, 48), 120).save(dossier_classe / "image.png")
        with self.assertRaisesRegex(ValueError, "7"):
            preparer_donnees(dossier_incomplet)


if __name__ == "__main__":
    unittest.main()
