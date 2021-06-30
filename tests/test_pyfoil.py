import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import os
from pyfoil.pyFoil import readCoordFile, Airfoil, _getClosestY
from pyfoil import sampling

baseDir = os.path.dirname(os.path.abspath(__file__))


class TestBasic(unittest.TestCase):
    def setUp(self):
        X = readCoordFile(os.path.join(baseDir, "airfoils/flat_plate.dat"))
        self.foil = Airfoil(X)
        self.chord = 1
        self.twist = 0

    def test_chord(self):
        assert_allclose(self.foil.getChord(), 1, atol=1e-10)

    def test_twist(self):
        assert_allclose(self.foil.getTwist(), 0, atol=1e-10)

    def test_findPt(self):
        assert_allclose(self.foil.findPt(0.8)[0][0], 0.8, atol=1e-10)

    def test_splitAirfoil(self):
        top, bottom = self.foil.splitAirfoil()
        assert_array_equal(top.getValue(0), np.array([1, 0]))
        assert_array_equal(top.getValue(1), np.array([0, 0]))
        assert_array_equal(bottom.getValue(0), np.array([0, 0]))
        assert_array_equal(bottom.getValue(1), np.array([1, 0]))

    def test_scale_basic(self):
        self.foil.scale(2)
        coords = self.foil.getSplinePts()
        assert_array_equal(coords, np.array([[2, 0], [0, 0], [2, 0]]))
        self.assertEqual(self.chord*2, self.foil.getChord())
        self.assertEqual(self.twist, self.foil.getTwist())

    def test_scale_off_center(self):
        self.foil.scale(2, origin=np.array([1, 0]))
        coords = self.foil.getSplinePts()
        assert_array_equal(coords, np.array([[1, 0], [-1, 0], [1, 0]]))
        self.assertEqual(self.chord*2, self.foil.getChord())
        self.assertEqual(self.twist, self.foil.getTwist())

    def test_normalizeChord(self):
        self.foil.scale(2)
        self.foil.normalizeChord()
        coords = self.foil.getSplinePts()
        assert_array_equal(coords, np.array([[1, 0], [0, 0], [1, 0]]))
        self.assertEqual(self.chord, self.foil.getChord())
        self.assertEqual(self.twist, self.foil.getTwist())

    def test_translate(self):
        self.foil.translate([1, -3])
        coords = self.foil.getSplinePts()
        assert_array_equal(coords, np.array([[2, -3], [1, -3], [2, -3]]))
        self.assertEqual(self.chord, self.foil.getChord())
        self.assertEqual(self.twist, self.foil.getTwist())

    def test_center(self):
        self.foil.translate([1, -3])
        self.foil.center()
        coords = self.foil.getSplinePts()
        assert_array_equal(coords, np.array([[1, 0], [0, 0], [1, 0]]))
        self.assertEqual(self.chord, self.foil.getChord())
        self.assertEqual(self.twist, self.foil.getTwist())

    def test_rotate_basic(self):
        self.foil.rotate(-45)
        coords = self.foil.getSplinePts()
        ref = np.array([[1 / np.sqrt(2), -1 / np.sqrt(2)], [0, 0], [1 / np.sqrt(2), -1 / np.sqrt(2)]])
        assert_allclose(coords, ref, atol=1e-10)
        self.assertAlmostEqual(self.chord, self.foil.getChord())
        self.assertAlmostEqual(self.twist-45, self.foil.getTwist())

    def test_rotate_off_center(self):
        self.foil.rotate(-45, [1, 0])
        coords = self.foil.getSplinePts()
        ref = np.array([[1, 0], [1 - 1 / np.sqrt(2), 1 / np.sqrt(2)], [1, 0]])
        assert_allclose(coords, ref, atol=1e-10)
        self.assertAlmostEqual(self.chord, self.foil.getChord())
        self.assertAlmostEqual(self.twist-45, self.foil.getTwist())

    def test_derotate(self):
        self.foil.rotate(45)
        self.foil.derotate()
        coords = self.foil.getSplinePts()
        assert_array_equal(coords, np.array([[1, 0], [0, 0], [1, 0]]))
        self.assertAlmostEqual(self.chord, self.foil.getChord())
        self.assertAlmostEqual(self.twist, self.foil.getTwist())

    def test_normalizeAirfoil(self):
        self.foil.rotate(15)
        self.foil.scale(30)
        self.foil.translate([2, 14])
        self.foil.rotate(-29)
        self.foil.normalizeAirfoil()
        coords = self.foil.getSplinePts()
        assert_allclose(coords, np.array([[1, 0], [0, 0], [1, 0]]), atol=1e-10)
        self.assertAlmostEqual(self.chord, self.foil.getChord())
        self.assertAlmostEqual(self.twist, self.foil.getTwist())


class TestSampling(unittest.TestCase):
    # for now these just test if it runs without error, not if the output is right
    def setUp(self):
        X = readCoordFile(os.path.join(baseDir, "airfoils/rae2822.dat"))
        self.foil = Airfoil(X)

    def test_defaults(self):
        self.foil.getSampledPts(100, nTEPts=10)

    def test_custom_dist_sample(self):
        self.foil.getSampledPts(100, spacingFunc=np.linspace)

    def test_pass_args_to_dist(self):
        func_args = {"coeff": 2}
        self.foil.getSampledPts(100, spacingFunc=sampling.conical, func_args=func_args)


class TestSamplingTE(unittest.TestCase):
    def setUp(self):
        X = readCoordFile(os.path.join(baseDir, "airfoils/hypersonic_glider.dat"))
        self.hg = Airfoil(X)

    def test_nTEPts(self):
        self.assertFalse(self.hg.closedCurve)
        coords = self.hg.getSampledPts(100, nTEPts=1)
        TEref = np.array([1, 0])
        assert_array_equal(TEref, coords[-2, :])

    def test_noTE_knot_noPts(self):
        self.assertFalse(self.hg.closedCurve)
        coords = self.hg.getSampledPts(100)
        ref_upper = np.array([1, 0.5])
        ref_lower = np.array([1, -0.5])
        assert_array_equal(coords[-1, :], ref_upper)
        assert_array_equal(coords[-2, :], ref_lower)
        self.assertNotEqual(coords[-2, 0], coords[-3, 0])
        self.assertNotEqual(coords[-2, 1], coords[-3, 1])

    def test_TE_knot_noPts(self):
        self.assertFalse(self.hg.closedCurve)
        coords = self.hg.getSampledPts(100, TE_knot=True)
        assert_array_equal(coords[-1, :], coords[0, :])
        assert_array_equal(coords[-2, :], coords[-3, :])

    def test_TE_knot(self):
        self.assertFalse(self.hg.closedCurve)
        coords = self.hg.getSampledPts(100, TE_knot=True, nTEPts=1)
        assert_array_equal(coords[-1, :], coords[0, :])
        assert_array_equal(coords[-3, :], coords[-4, :])



class TestGeoModification(unittest.TestCase):
    def setUp(self):
        X = readCoordFile(os.path.join(baseDir, "airfoils/rae2822.dat"))
        self.foil = Airfoil(X)

    def test_reorder(self):
        Xorig = self.foil.spline.X
        newfoil = Airfoil(Xorig[::-1, :])
        Xreordered = newfoil.spline.X
        assert_allclose(Xorig, Xreordered, atol=1e-6)

    def test_round_TE(self):
        self.foil.roundTE(k=4)
        refTE = np.array([0.990393, 0.0013401])
        newTE = self.foil.TE
        assert_allclose(refTE, newTE, atol=1e-6)
        self.assertTrue(self.foil.closedCurve)

    def test_blunt_TE(self):
        self.foil.makeBluntTE()
        refTE = np.array([0.97065494, 0.00352594])
        newTE = self.foil.TE
        assert_allclose(refTE, newTE, atol=1e-8)
        self.assertFalse(self.foil.closedCurve)


class TestFFD(unittest.TestCase):
    def setUp(self):
        X = readCoordFile(os.path.join(baseDir, "airfoils/rae2822.dat"))
        self.foil = Airfoil(X)
        X = readCoordFile(os.path.join(baseDir, "airfoils/wave.dat"))
        self.wave = Airfoil(X)

    def test_getClosest(self):
        yu, yl = _getClosestY(self.wave.getSplinePts(), 0.3)
        self.assertEqual(yu, 0.25)
        self.assertEqual(yl, -0.25)

    def test_getClosest_off_end(self):
        yu, yl = _getClosestY(self.wave.getSplinePts(), -1)
        self.assertEqual(yu, 0)
        self.assertEqual(yl, 0)

    def test_box_FFD(self):
        FFD_box = self.foil._buildFFD(4, False, 0.001, 0.02, 0.02, None, None)
        FFD_box_actual = np.zeros((4, 2, 2, 3))

        xslice = np.array([-0.001, 0.3297966667, 0.6605933334, 0.9913900001])
        FFD_box_actual[:, 0, 0, 0] = xslice[:].copy()
        FFD_box_actual[:, 1, 0, 0] = xslice[:].copy()
        FFD_box_actual[:, 0, 0, 1] = -0.059236 - 0.02
        FFD_box_actual[:, 1, 0, 1] = 0.062779 + 0.02
        FFD_box_actual[:, :, 1, :] = FFD_box_actual[:, :, 0, :].copy()
        FFD_box_actual[:, :, 0, 2] = 0.0
        FFD_box_actual[:, :, 1, 2] = 1.0

        assert_allclose(FFD_box, FFD_box_actual, atol=1e-5)

    def test_fitted_FFD(self):
        FFD_box = self.wave._buildFFD(3, True, 0.001, 0.02, 0.02, None, None)

        FFD_box_actual = np.zeros((3, 2, 2, 3))
        xslice = np.array([-0.001, 0.5, 1.001])
        FFD_box_actual[:, 0, 0, 0] = xslice[:].copy()
        FFD_box_actual[:, 1, 0, 0] = xslice[:].copy()
        FFD_box_actual[0, 0, 0, 1] = -0.02
        FFD_box_actual[1, 0, 0, 1] = -0.27
        FFD_box_actual[2, 0, 0, 1] = -0.02
        FFD_box_actual[0, 1, 0, 1] = 0.02
        FFD_box_actual[1, 1, 0, 1] = 0.27
        FFD_box_actual[2, 1, 0, 1] = 0.02
        FFD_box_actual[:, :, 1, :] = FFD_box_actual[:, :, 0, :].copy()
        FFD_box_actual[:, :, 0, 2] = 0.0
        FFD_box_actual[:, :, 1, 2] = 1.0

        assert_allclose(FFD_box, FFD_box_actual, atol=1e-8)

    def test_specify_coord(self):
        coords = self.wave.getSplinePts()
        FFD_box = self.foil._buildFFD(3, True, 0.001, 0.02, 0.02, None, coords)

        FFD_box_actual = np.zeros((3, 2, 2, 3))
        xslice = np.array([-0.001, 0.5, 1.001])
        FFD_box_actual[:, 0, 0, 0] = xslice[:].copy()
        FFD_box_actual[:, 1, 0, 0] = xslice[:].copy()
        FFD_box_actual[0, 0, 0, 1] = -0.02
        FFD_box_actual[1, 0, 0, 1] = -0.27
        FFD_box_actual[2, 0, 0, 1] = -0.02
        FFD_box_actual[0, 1, 0, 1] = 0.02
        FFD_box_actual[1, 1, 0, 1] = 0.27
        FFD_box_actual[2, 1, 0, 1] = 0.02
        FFD_box_actual[:, :, 1, :] = FFD_box_actual[:, :, 0, :].copy()
        FFD_box_actual[:, :, 0, 2] = 0.0
        FFD_box_actual[:, :, 1, 2] = 1.0

        assert_allclose(FFD_box, FFD_box_actual, atol=1e-8)

    def test_specify_xslice(self):
        xslice = np.array([0.0, 0.5, 1.0])
        FFD_box = self.wave._buildFFD(7, True, 0.001, 0.02, 0.02, xslice, None)

        FFD_box_actual = np.zeros((3, 2, 2, 3))
        FFD_box_actual[:, 0, 0, 0] = xslice[:].copy()
        FFD_box_actual[:, 1, 0, 0] = xslice[:].copy()
        FFD_box_actual[0, 0, 0, 1] = -0.02
        FFD_box_actual[1, 0, 0, 1] = -0.27
        FFD_box_actual[2, 0, 0, 1] = -0.02
        FFD_box_actual[0, 1, 0, 1] = 0.02
        FFD_box_actual[1, 1, 0, 1] = 0.27
        FFD_box_actual[2, 1, 0, 1] = 0.02
        FFD_box_actual[:, :, 1, :] = FFD_box_actual[:, :, 0, :].copy()
        FFD_box_actual[:, :, 0, 2] = 0.0
        FFD_box_actual[:, :, 1, 2] = 1.0

        assert_allclose(FFD_box, FFD_box_actual, atol=1e-8)


class TestCamber(unittest.TestCase):
    def setUp(self):
        self.foil = Airfoil(readCoordFile(os.path.join(baseDir, "airfoils/rae2822.dat")))

    def test_rae2822_thickness(self):
        maxThickness = self.foil.getMaxThickness("british")
        np.testing.assert_allclose(maxThickness, [0.379, 0.121], rtol=1e-2)

    def test_rae2822_camber(self):
        maxCamber = self.foil.getMaxCamber()
        np.testing.assert_allclose(maxCamber, [0.757, 0.013], rtol=0.15)


if __name__ == "__main__":
    unittest.main()
