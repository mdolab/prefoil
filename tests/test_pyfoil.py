import unittest
import numpy as np
import os
from pyfoil.pyFoil import readCoordFile, Airfoil
from pyfoil import sampling

baseDir = os.path.dirname(os.path.abspath(__file__))


class TestBasic(unittest.TestCase):
    def setUp(self):
        X = readCoordFile(os.path.join(baseDir, "testAirfoil.dat"))
        self.foil = Airfoil(X)

    def test_rotate(self):
        self.foil.rotate(45)

        np.testing.assert_array_almost_equal(self.foil.TE, np.sqrt(2) / 2 * np.ones(2))

        self.foil.derotate()

        np.testing.assert_array_almost_equal(self.foil.TE, np.array([1, 0]))

    def test_chord(self):
        self.assertEqual(self.foil.getChord(), 1)

    def test_twist(self):
        self.assertEqual(self.foil.getTwist(), 0)

    def test_findPt(self):
        self.assertAlmostEqual(self.foil.findPt(0.8)[0][0], 0.8)


class TestSampling(unittest.TestCase):
    # for now these just test if it runs without error, not if the output is right
    def setUp(self):
        X = readCoordFile(os.path.join(baseDir, "rae2822.dat"))
        self.foil = Airfoil(X)

    def test_defaults(self):
        self.foil.getSampledPts(100, nTEPts=10)

    def test_custom_dist_sample(self):
        self.foil.getSampledPts(100, spacingFunc=np.linspace)

    def test_pass_args_to_dist(self):
        func_args = {"coeff": 2}
        self.foil.getSampledPts(100, spacingFunc=sampling.conical, func_args=func_args)


class TestGeoModification(unittest.TestCase):
    def setUp(self):
        X = readCoordFile(os.path.join(baseDir, "rae2822.dat"))
        self.foil = Airfoil(X)

    def test_reorder(self):
        Xorig = self.foil.spline.X
        newfoil = Airfoil(Xorig[::-1, :])
        Xreordered = newfoil.spline.X
        np.testing.assert_array_almost_equal(Xorig, Xreordered, decimal=6)

    def test_round_TE(self):
        self.foil.roundTE(k=4)
        refTE = np.array([0.990393, 0.0013401])
        newTE = self.foil.TE
        np.testing.assert_array_almost_equal(refTE, newTE, decimal=6)

    def test_blunt_TE(self):
        self.foil.makeBluntTE()
        refTE = np.array([0.97065494, 0.00352594])
        newTE = self.foil.TE
        np.testing.assert_array_almost_equal(refTE, newTE, decimal=8)


if __name__ == "__main__":
    unittest.main()
