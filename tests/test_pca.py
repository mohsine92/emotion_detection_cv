"""Vérifications rapides de la PCA, sans télécharger FER2013."""

import unittest

import matplotlib

# Les tests créent les figures en mémoire, sans ouvrir de fenêtre.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.emotion_detection.pca import (
    TAILLE_VECTEUR,
    afficher_composantes,
    afficher_projection_2d,
    afficher_reconstructions,
    comparer_nombres_composantes,
    composantes_pour_variance,
    creer_pipeline_pca,
    etudier_variance,
    reduire_donnees,
    tracer_comparaison,
    tracer_variance,
)


def creer_donnees(nombre, graine):
    """Images fictives : un motif propre à chaque classe + du bruit, dans [0, 1]."""
    generateur = np.random.default_rng(graine)
    motifs = np.random.default_rng(0).uniform(0.2, 0.8, size=(3, TAILLE_VECTEUR))
    y = np.arange(nombre) % 3
    X = motifs[y] + generateur.normal(0, 0.05, size=(nombre, TAILLE_VECTEUR))
    return np.clip(X, 0, 1).astype(np.float32), y


class TestReductionPCA(unittest.TestCase):
    """Petits ensembles fabriqués ; 5 composantes suffisent pour les tests."""

    def setUp(self):
        self.addCleanup(plt.close, "all")
        self.X_train, self.y_train = creer_donnees(60, graine=1)
        self.X_val, self.y_val = creer_donnees(20, graine=2)
        self.X_test, self.y_test = creer_donnees(25, graine=3)

    def test_formes_des_sorties(self):
        pca, X_train_pca, X_val_pca, X_test_pca = reduire_donnees(
            self.X_train, self.X_val, self.X_test, n_composantes=5
        )
        self.assertEqual(X_train_pca.shape, (60, 5))
        self.assertEqual(X_val_pca.shape, (20, 5))
        self.assertEqual(X_test_pca.shape, (25, 5))
        self.assertEqual(pca.components_.shape, (5, TAILLE_VECTEUR))
        for X in (X_train_pca, X_val_pca, X_test_pca):
            self.assertTrue(np.isfinite(X).all())

    def test_pca_ajustee_sur_le_train_uniquement(self):
        pca, _, X_val_pca, X_test_pca = reduire_donnees(
            self.X_train, self.X_val, self.X_test, n_composantes=5
        )
        # La moyenne retirée par la PCA est celle du train, pas celle de val/test.
        self.assertEqual(pca.n_samples_, len(self.X_train))
        np.testing.assert_allclose(pca.mean_, self.X_train.mean(axis=0), atol=1e-5)

        # Val et test sont seulement projetés avec la PCA du train.
        np.testing.assert_allclose(
            X_val_pca, (self.X_val - pca.mean_) @ pca.components_.T, atol=1e-4
        )
        np.testing.assert_allclose(
            X_test_pca, (self.X_test - pca.mean_) @ pca.components_.T, atol=1e-4
        )

    def test_validation_et_test_ne_changent_pas_la_pca(self):
        pca_1, train_1, _, _ = reduire_donnees(
            self.X_train, self.X_val, self.X_test, n_composantes=5
        )
        # On remplace complètement val et test : la PCA doit rester identique.
        autre_val, _ = creer_donnees(20, graine=50)
        autre_test, _ = creer_donnees(25, graine=60)
        pca_2, train_2, _, _ = reduire_donnees(
            self.X_train, autre_val, autre_test, n_composantes=5
        )
        np.testing.assert_allclose(pca_1.components_, pca_2.components_, atol=1e-6)
        np.testing.assert_allclose(train_1, train_2, atol=1e-5)

    def test_reduction_reproductible(self):
        premier = reduire_donnees(self.X_train, self.X_val, self.X_test, 5)
        second = reduire_donnees(self.X_train, self.X_val, self.X_test, 5)
        for a, b in zip(premier[1:], second[1:]):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_composantes_orthonormees(self):
        pca, *_ = reduire_donnees(self.X_train, self.X_val, self.X_test, 5)
        np.testing.assert_allclose(
            pca.components_ @ pca.components_.T, np.eye(5), atol=1e-4
        )

    def test_entrees_invalides_refusees(self):
        with self.assertRaises(ValueError):  # images non aplaties
            reduire_donnees(self.X_train[:, :100], self.X_val, self.X_test, 5)
        with self.assertRaises(ValueError):  # trop de composantes
            reduire_donnees(self.X_train, self.X_val, self.X_test, 61)
        with self.assertRaises(ValueError):  # zéro composante
            reduire_donnees(self.X_train, self.X_val, self.X_test, 0)
        X_val_nan = self.X_val.copy()
        X_val_nan[0, 0] = np.nan
        with self.assertRaises(ValueError):  # valeur manquante
            reduire_donnees(self.X_train, X_val_nan, self.X_test, 5)


class TestEtudeVariance(unittest.TestCase):
    """La variance cumulée sert à choisir le nombre de composantes."""

    def setUp(self):
        self.addCleanup(plt.close, "all")
        self.X_train, self.y_train = creer_donnees(60, graine=1)
        self.pca_complete, self.variance = etudier_variance(self.X_train)

    def test_variance_cumulee_croissante_jusqu_a_un(self):
        self.assertTrue(np.all(np.diff(self.variance) >= -1e-9))
        # Images en float32 : la somme vaut 1 à un petit arrondi près.
        self.assertAlmostEqual(float(self.variance[-1]), 1.0, places=4)
        self.assertTrue(np.all((self.variance > 0) & (self.variance <= 1 + 1e-4)))

    def test_nombre_de_composantes_pour_un_seuil(self):
        variance = np.array([0.5, 0.8, 0.9, 0.95, 1.0])
        self.assertEqual(composantes_pour_variance(variance, 0.5), 1)
        self.assertEqual(composantes_pour_variance(variance, 0.85), 3)
        self.assertEqual(composantes_pour_variance(variance, 0.95), 4)
        self.assertEqual(composantes_pour_variance(variance, 1.0), 5)
        with self.assertRaises(ValueError):
            composantes_pour_variance(variance, 0)
        with self.assertRaises(ValueError):
            composantes_pour_variance(variance, 1.5)

    def test_figures_creees(self):
        figure = tracer_variance(self.variance, n_choisi=5, plage=(3, 8))
        self.assertEqual(len(figure.axes), 1)
        self.assertGreaterEqual(len(figure.axes[0].lines), 2)

        figure = afficher_composantes(self.pca_complete, nb_composantes=16)
        visibles = [axe for axe in figure.axes if axe.get_visible()]
        self.assertEqual(len(visibles), 16)

        figure = afficher_reconstructions(self.pca_complete, self.X_train,
                                          nombres=(2, 5, 10), nb_images=2)
        self.assertEqual(len(figure.axes), 2 * 4)
        self.assertTrue(all(len(axe.images) == 1 for axe in figure.axes))

        _, X_train_pca, _, _ = reduire_donnees(self.X_train, self.X_train,
                                               self.X_train, 2)
        figure = afficher_projection_2d(X_train_pca, self.y_train,
                                        {"a": 0, "b": 1, "c": 2})
        self.assertEqual(len(figure.axes[0].collections), 3)


class TestComparaisonEtPipeline(unittest.TestCase):
    """Aide au choix du nombre de composantes et Pipeline pour la mission 05."""

    def setUp(self):
        self.addCleanup(plt.close, "all")
        self.X_train, self.y_train = creer_donnees(60, graine=1)
        self.X_val, self.y_val = creer_donnees(20, graine=2)

    def test_comparaison_des_nombres_de_composantes(self):
        resultats = comparer_nombres_composantes(
            self.X_train, self.y_train, self.X_val, self.y_val, nombres=(2, 5, 10)
        )
        self.assertEqual([r["composantes"] for r in resultats], [2, 5, 10])
        for r in resultats:
            self.assertTrue(0 <= r["accuracy_train"] <= 1)
            self.assertTrue(0 <= r["accuracy_val"] <= 1)
        variances = [r["variance"] for r in resultats]
        self.assertEqual(variances, sorted(variances))

        figure = tracer_comparaison(resultats)
        self.assertEqual(len(figure.axes[0].lines), 2)

    def test_pipeline_recoit_les_pixels_bruts(self):
        # Peu d'images ici : le solveur "full" est plus rapide pour ce test.
        pipeline = creer_pipeline_pca(n_composantes=5, solveur="full")
        pipeline.fit(self.X_train, self.y_train)

        self.assertEqual(pipeline.named_steps["pca"].n_components_, 5)
        probabilites = pipeline.predict_proba(self.X_val)
        self.assertEqual(probabilites.shape, (20, 3))
        np.testing.assert_allclose(probabilites.sum(axis=1), 1.0, atol=1e-6)
        self.assertEqual(pipeline.predict(self.X_val).shape, (20,))


if __name__ == "__main__":
    unittest.main()
