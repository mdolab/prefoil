from __future__ import print_function, division, absolute_import

import unittest
import numpy as np

from pyfoil import pyFoil
from pyfoil import sampling


class TestBasic(unittest.TestCase):

    def setUp(self):
        X = pyFoil._readCoordFile('testAirfoil.dat')
        self.foil = pyFoil.Airfoil(X)

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
        X = pyFoil._readCoordFile('rae2822.dat')
        self.foil = pyFoil.Airfoil(X)

    def test_defaults(self):
        self.foil.getSampledPts(100, nTEPts=10)

    def test_custom_dist_sample(self):
        self.foil.getSampledPts(100, spacingFunc=np.linspace)

    def test_pass_args_to_dist(self):
        func_args = {
            'coeff': 2
        }
        self.foil.getSampledPts(100, spacingFunc=sampling.conical, func_args=func_args)

class TestGeoModification(unittest.TestCase):

    def setUp(self):
        X = pyFoil._readCoordFile('rae2822.dat')
        self.foil = pyFoil.Airfoil(X)

    def test_round_TE(self):
        oldTE = self.foil.TE
        self.foil.roundTE(k=4)
        # the reason for the 4 decimal is because it doesn't exactly place the new TE point at the midpoint of the previous blunt foil
        np.testing.assert_array_almost_equal(oldTE, self.foil.spline.X[0], decimal=4)


if __name__ == '__main__':
    unittest.main()
