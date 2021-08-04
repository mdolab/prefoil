import unittest
from baseclasses import BaseRegTest
import os
from prefoil.preFoil import Airfoil, readCoordFile

baseDir = os.path.dirname(os.path.abspath(__file__))


class TestTEModification(unittest.TestCase):
    def setUp(self):
        X = readCoordFile(os.path.join(baseDir, "airfoils/rae2822.dat"))
        self.foil = Airfoil(X)

    def train_bluntTE(self):
        self.test_bluntTE(True)

    def test_bluntTE(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_bluntTE.ref")
        with BaseRegTest(ref_file, train) as handler:
            self.foil.makeBluntTE(0.75)
            handler.root_add_val("test_bluntTE - Spline Coords:", self.foil.spline.X, tol=1e-10)
            self.assertFalse(self.foil.closedCurve)

    def train_bluntTE_rotated(self):
        self.test_bluntTE_rotated(True)

    def test_bluntTE_rotated(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_bluntTE_rotated.ref")
        with BaseRegTest(ref_file, train) as handler:
            self.foil.rotate(-30)
            self.foil.makeBluntTE(0.75)
            handler.root_add_val("test_bluntTE_rotated - Spline Coords:", self.foil.spline.X, tol=1e-10)
            self.assertFalse(self.foil.closedCurve)

    def train_roundTE(self):
        self.test_roundTE(True)

    def test_roundTE(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_roundTE.ref")
        with BaseRegTest(ref_file, train) as handler:
            self.foil.roundTE(k=4)
            handler.root_add_val("test_roundTE - Spline Coords:", self.foil.spline.X, tol=1e-10)
            self.assertTrue(self.foil.closedCurve)

    def train_roundTE_rotated(self):
        self.test_roundTE_rotated(True)

    def test_roundTE_rotated(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_roundTE_rotated.ref")
        with BaseRegTest(ref_file, train) as handler:
            self.foil.rotate(-20)
            self.foil.roundTE(k=4)
            handler.root_add_val("test_roundTE_rotated - Spline Coords:", self.foil.spline.X, tol=1e-10)
            self.assertTrue(self.foil.closedCurve)

    def train_sharpenTE(self):
        self.test_sharpenTE(True)

    def test_sharpenTE(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_sharpenTE.ref")
        with BaseRegTest(ref_file, train) as handler:
            self.foil.sharpenTE()
            handler.root_add_val("test_sharpenTE - Spline Coords:", self.foil.spline.X, tol=1e-10)
            self.assertTrue(self.foil.closedCurve)

    def train_sharpenTE_rotated(self):
        self.test_sharpenTE_rotated(True)

    def test_sharpenTE_rotated(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_sharpenTE_rotated.ref")
        with BaseRegTest(ref_file, train) as handler:
            self.foil.rotate(10)
            self.foil.sharpenTE()
            handler.root_add_val("test_sharpenTE_rotated - Spline Coords:", self.foil.spline.X, tol=1e-10)
            self.assertTrue(self.foil.closedCurve)
