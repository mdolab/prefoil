import unittest
from baseclasses import BaseRegTest
import os
from pyfoil import sampling

baseDir = os.path.dirname(os.path.abspath(__file__))


class TestBasic(unittest.TestCase):
    def train_cosine(self):
        self.test_cosine(True)

    def test_cosine(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_cosine.ref")
        with BaseRegTest(ref_file, train) as handler:
            s = sampling.cosine(0, 1, 100)
            handler.root_add_val("test_cosine - Sample from Cosine:", s, tol=1e-10)

    def train_polynomial(self):
        self.test_polynomial(True)

    def test_polynomial(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_polynomial.ref")
        with BaseRegTest(ref_file, train) as handler:
            s = sampling.polynomial(0, 1, 100)
            handler.root_add_val("test_polynomial - Sample from Polynomial:", s, tol=1e-10)

    def train_conical(self):
        self.test_conical(True)

    def test_conical(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_conical.ref")
        with BaseRegTest(ref_file, train) as handler:
            s = sampling.conical(0, 1, 100, coeff=2)
            handler.root_add_val("test_conical - Sample from Conical with coeff = 2:", s, tol=1e-10)

    def train_bigeometric(self):
        self.test_bigeometric(True)

    def test_bigeometric(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_bigeometric.ref")
        with BaseRegTest(ref_file, train) as handler:
            s = sampling.bigeometric(0, 1, 100)
            handler.root_add_val("test_bigeometric - Sample from Bigeometric:", s, tol=1e-10)
