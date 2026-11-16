"""Vérifications rapides de la conversion NumPy, sans télécharger FER2013."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from src.emotion_detection.conversion import (
    TAILLE_VECTEUR,
    convertir_donnees,
    dataset_vers_numpy,
    verifier_numpy,
)
from src.emotion_detection.preprocessing import preparer_donnees


def creer_ensemble(nombre):
    """Image i = gris uniforme i/nombre, avec le label i."""
    labels = torch.arange(nombre)
    valeurs = labels.float() / nombre
    images = valeurs.view(nombre, 1, 1, 1).expand(nombre, 1, 48, 48).clone()
    return TensorDataset(images, labels)


class TestConversionNumpy(unittest.TestCase):
    """Petits lots fabriqués pour le test."""

    def test_formes_et_types(self):
        X, y = dataset_vers_numpy(creer_ensemble(10), taille_lot=4)

        self.assertEqual(X.shape, (10, TAILLE_VECTEUR))
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.shape, (10,))
        self.assertTrue(np.issubdtype(y.dtype, np.integer))
        self.assertTrue(np.isfinite(X).all())
        self.assertGreaterEqual(X.min(), 0.0)
        self.assertLessEqual(X.max(), 1.0)

    def test_association_image_label_conservee(self):
        n = 10
        X, y = dataset_vers_numpy(creer_ensemble(n), taille_lot=3)

        np.testing.assert_array_equal(y, np.arange(n))
        attendu = (torch.arange(n).float() / n).numpy()
        np.testing.assert_array_equal(X[:, 0], attendu)
        # Chaque image est uniforme : tous ses pixels valent le premier.
        self.assertTrue(np.all(X == X[:, :1]))

    def test_aplatissement_ligne_par_ligne(self):
        image = torch.zeros(1, 1, 48, 48)
        image[0, 0, 1, 2] = 1.0  # ligne 1, colonne 2

        X, _ = dataset_vers_numpy(TensorDataset(image, torch.tensor([0])))

        self.assertEqual(X[0, 1 * 48 + 2], 1.0)
        self.assertEqual(X.sum(), 1.0)

    def test_resultat_independant_de_la_taille_de_lot(self):
        ensemble = creer_ensemble(10)

        X1, y1 = dataset_vers_numpy(ensemble, taille_lot=1)
        X2, y2 = dataset_vers_numpy(ensemble, taille_lot=256)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_ensemble_vide_refuse(self):
        vide = TensorDataset(
            torch.empty(0, 1, 48, 48), torch.empty(0, dtype=torch.long)
        )
        with self.assertRaises(ValueError):
            dataset_vers_numpy(vide)


class TestVerification(unittest.TestCase):
    """verifier_numpy accepte les bonnes données et refuse les autres."""

    def setUp(self):
        self.X = np.full((4, TAILLE_VECTEUR), 0.5, dtype=np.float32)
        self.y = np.arange(4)

    def test_donnees_valides_acceptees(self):
        verifier_numpy(self.X, self.y)

    def test_valeurs_invalides_refusees(self):
        X_nan = self.X.copy()
        X_nan[0, 0] = np.nan
        X_inf = self.X.copy()
        X_inf[0, 0] = np.inf
        X_grand = self.X.copy()
        X_grand[0, 0] = 1.5
        X_negatif = self.X.copy()
        X_negatif[0, 0] = -0.1

        for nom, X in [("nan", X_nan), ("inf", X_inf),
                       ("trop grand", X_grand), ("négatif", X_negatif)]:
            with self.subTest(nom):
                with self.assertRaises(ValueError):
                    verifier_numpy(X, self.y)

    def test_formes_invalides_refusees(self):
        cas = {
            "colonnes": (self.X[:, :100], self.y),
            "y trop court": (self.X, self.y[:3]),
            "y en 2D": (self.X, self.y.reshape(4, 1)),
            "X vide": (np.empty((0, TAILLE_VECTEUR), dtype=np.float32),
                       np.empty((0,), dtype=np.int64)),
        }
        for nom, (X, y) in cas.items():
            with self.subTest(nom):
                with self.assertRaises(ValueError):
                    verifier_numpy(X, y)

    def test_types_invalides_refuses(self):
        with self.assertRaises(TypeError):
            verifier_numpy(self.X.astype(np.int64), self.y)
        with self.assertRaises(TypeError):
            verifier_numpy(self.X, self.y.astype(np.float32))


class TestIntegrationPretraitement(unittest.TestCase):
    """Les sorties de la mission 02 passent bien dans la mission 03."""

    def test_pipeline_avec_la_mission_02(self):
        classes = ["angry", "disgust", "fear", "happy",
                   "neutral", "sad", "surprise"]  # ordre alphabétique

        with tempfile.TemporaryDirectory() as dossier_temporaire:
            dossier = Path(dossier_temporaire) / "data" / "train"
            for indice, nom in enumerate(classes):
                (dossier / nom).mkdir(parents=True)
                for numero in range(4):
                    Image.new("L", (48, 48), 30 + indice * 20).save(
                        dossier / nom / f"image_{numero}.png"
                    )

            train_ds, val_ds, test_ds, _ = preparer_donnees(dossier)
            X_train, y_train, X_val, y_val, X_test, y_test = convertir_donnees(
                train_ds, val_ds, test_ds
            )

        self.assertEqual((len(y_train), len(y_val), len(y_test)), (19, 4, 5))
        for X, y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]:
            self.assertEqual(X.shape, (len(y), TAILLE_VECTEUR))
            # La couleur connue de chaque classe doit rester liée à son label.
            np.testing.assert_allclose(X[:, 0], (30 + y * 20) / 255, atol=1e-6)


if __name__ == "__main__":
    unittest.main()