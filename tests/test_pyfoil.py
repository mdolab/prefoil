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


class TestSampling(unittest.TestCase):

    def setUp(self):
        X = pyFoil._readCoordFile('rae2822.dat')
        self.foil = pyFoil.Airfoil(X)

    def test_defaults(self):

        self.foil.sample(100)
        # self.foil.plot()

    def test_custom_dist_sample(self):
        self.foil.sample(100, spacingFunc=np.linspace)

    def test_pass_args_to_dist(self):
        # here
        func_args = {
            'coeff': 2
        }
        self.foil.sample(100, spacingFunc=sampling.conical, func_args=func_args)
        # self.foil.plot()

    # def


if __name__ == '__main__':
    # unittest.main()
    X = pyFoil._readCoordFile('testAirfoil.dat')
    foil = pyFoil.Airfoil(X)
    # foil.rotate(45)
    foil.sample()
    foil.plot()
    foil.derotate()
    print(foil.TE, foil.LE)
