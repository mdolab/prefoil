import unittest
from baseclasses import BaseRegTest
import os
from pyfoil import sampling

baseDir = os.path.dirname(os.path.abspath(__file__))


class TestBasic(unittest.TestCase):
    def train_cosine(self, train=True):
        try:
            self.test_cosine(train=train)
        except AttributeError:
            raise unittest.SkipTest()

    def test_cosine(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_cosine.ref")
        with BaseRegTest(ref_file, train=train) as handler:
            s = sampling.cosine(0, 1, 100)
            handler.root_add_val("{test_cosine} - Sample from Cosine:", s, tol=1e-10)

    def train_polynomial(self, train=True):
        try:
            self.test_polynomial(train=train)
        except AttributeError:
            raise unittest.SkipTest()

    def test_polynomial(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_polynomial.ref")
        with BaseRegTest(ref_file, train=train) as handler:
            s = sampling.polynomial(0, 1, 100)
            handler.root_add_val("{test_polynomial} - Sample from Polynomial:", s, tol=1e-10)

    def train_conical(self, train=True):
        try:
            self.test_conical(train=train)
        except AttributeError:
            raise unittest.SkipTest()

    def test_conical(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_conical.ref")
        with BaseRegTest(ref_file, train=train) as handler:
            s = sampling.conical(0, 1, 100, coeff=2)
            handler.root_add_val("{test_conical} - Sample from Conical with coeff = 2:", s, tol=1e-10)

    def train_bigeometic(self, train=True):
        try:
            self.test_bigeometic(train=train)
        except AttributeError:
            raise unittest.SkipTest()

    def test_bigeometic(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_bigeometic.ref")
        with BaseRegTest(ref_file, train=train) as handler:
            s = sampling.bigeometric(0, 1, 100)
            handler.root_add_val("{test_bigeometic} - Sample from Bigeometric:", s, tol=1e-10)
