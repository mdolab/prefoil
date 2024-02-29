import unittest
from baseclasses import BaseRegTest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import os
from prefoil import Airfoil, sampling
from prefoil.utils import readCoordFile, Error, generateNACA
from prefoil.utils.geom_ops import _getClosestY

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
        self.assertEqual(self.chord * 2, self.foil.getChord())
        self.assertEqual(self.twist, self.foil.getTwist())

    def test_scale_off_center(self):
        self.foil.scale(2, origin=np.array([1, 0]))
        coords = self.foil.getSplinePts()
        assert_array_equal(coords, np.array([[1, 0], [-1, 0], [1, 0]]))
        self.assertEqual(self.chord * 2, self.foil.getChord())
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
        self.assertAlmostEqual(self.twist - 45, self.foil.getTwist())

    def test_rotate_off_center(self):
        self.foil.rotate(-45, [1, 0])
        coords = self.foil.getSplinePts()
        ref = np.array([[1, 0], [1 - 1 / np.sqrt(2), 1 / np.sqrt(2)], [1, 0]])
        assert_allclose(coords, ref, atol=1e-10)
        self.assertAlmostEqual(self.chord, self.foil.getChord())
        self.assertAlmostEqual(self.twist - 45, self.foil.getTwist())

    def test_derotate(self):
        self.foil.rotate(45)
        self.foil.derotate()
        coords = self.foil.getSplinePts()
        assert_allclose(coords, np.array([[1, 0], [0, 0], [1, 0]]), atol=1e-14)
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

    def test_chordProj_basic(self):
        val = self.foil._findChordProj(np.array([0.57, 2]))
        assert_allclose(np.array([0.57, 0]), val, atol=1e-12)

    def test_chordProj_vertical(self):
        self.foil.rotate(-90)
        val = self.foil._findChordProj(np.array([3.7, -0.76]))
        assert_allclose(np.array([0, -0.76]), val, atol=1e-12)

    def test_chordProj_angle(self):
        self.foil.rotate(-30)
        val = self.foil._findChordProj(np.array([0.5, 0]))
        assert_allclose(np.array([3 / 8, -np.sqrt(3) / 8]), val, atol=1e-12)

    def test_generateNACA_0012(self):
        af = Airfoil(generateNACA("0012", 200))
        self.assertFalse(af.closedCurve)
        assert_allclose(np.array([0.0, 0.0]), af.LE, atol=1e-12)
        assert_allclose(np.array([1.0, 0.0]), af.TE, atol=1e-12)
        assert_allclose(0.00126 * 2, af.getTEThickness(), atol=1e-12)
        self.assertTrue(af.isSymmetric())
        assert_allclose((0.30, 0.12), af.getMaxThickness("american"), atol=1e-4)

    def test_generateNACA_6412(self):
        af = Airfoil(generateNACA("6412", 1000))
        self.assertFalse(af.closedCurve)
        assert_allclose(np.array([1.0, 0.0]), af.TE, atol=1e-12)
        self.assertFalse(af.isSymmetric())
        assert_allclose((0.30, 0.12), af.getMaxThickness("american"), atol=1e-4)
        assert_allclose((0.396, 0.06), af.getMaxCamber(), atol=3e-2)

    def test_generateNACA_code(self):
        with self.assertRaises(Error):
            generateNACA("90111", 200)

    def test_generateNACA_code_digit(self):
        with self.assertRaises(Error):
            generateNACA("9s11", 200)


class TestSampling(unittest.TestCase):
    def setUp(self):
        X = readCoordFile(os.path.join(baseDir, "airfoils/rae2822.dat"))
        self.foil = Airfoil(X)

    def train_defaults(self, train=True):
        self.test_defaults(train=train)

    def test_defaults(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_defaults.ref")
        with BaseRegTest(ref_file, train=train) as handler:
            points = self.foil.getSampledPts(100, nTEPts=10)
            handler.root_add_val("test_default - Default RAE2822 sampled points:", points, tol=1e-10)

    def train_linspace(self, train=True):
        self.test_linspace(train=train)

    def test_linspace(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_linspace.ref")
        with BaseRegTest(ref_file, train=train) as handler:
            points = self.foil.getSampledPts(100, spacingFunc=np.linspace)
            handler.root_add_val("test_linspace - Linear RAE2822 sampled points:", points, tol=1e-10)

    def train_pass_args_to_dist(self, train=True):
        self.test_pass_args_to_dist(train=train)

    def test_pass_args_to_dist(self, train=False):
        ref_file = os.path.join(baseDir, "ref/test_args.ref")
        with BaseRegTest(ref_file, train=train) as handler:
            points = self.foil.getSampledPts(100, spacingFunc=sampling.conical, func_args={"coeff": 2})
            handler.root_add_val("{test_args} - Conical RAE2822 sampled points:", points, tol=1e-10)


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
        assert_allclose(maxThickness, [0.379, 0.121], rtol=1e-2)

    def test_rae2822_camber(self):
        maxCamber = self.foil.getMaxCamber()
        assert_allclose(maxCamber, [0.757, 0.013], rtol=0.1)

    def test_rae2822_camber_scale(self):
        self.foil.scale(1.5)
        maxCamber = self.foil.getMaxCamber()
        assert_allclose(maxCamber, [0.757, 0.013], rtol=0.1)

    def test_rae2822_camber_translate(self):
        self.foil.translate([0.2, -0.23])
        maxCamber = self.foil.getMaxCamber()
        assert_allclose(maxCamber, [0.757, 0.013], rtol=0.1)

    def test_rae2822_camber_rot(self):
        self.foil.rotate(-30)
        maxCamber = self.foil.getMaxCamber()
        assert_allclose(maxCamber, [0.757, 0.013], rtol=0.1)


class TestFileWriting(unittest.TestCase):
    def setUp(self):
        self.foil = Airfoil(readCoordFile(os.path.join(baseDir, "airfoils/rae2822.dat")))
        self.temp_ffd = os.path.join(baseDir, "writeFFD_temp")
        self.temp_p3d = os.path.join(baseDir, "writeP3D_temp")
        self.temp_dat = os.path.join(baseDir, "writeDat_temp")

    def test_writeFFD(self):
        self.foil.generateFFD(10, self.temp_ffd)
        self.temp_ffd += ".xyz"
        with open(os.path.join(baseDir, "ref/rae2822_ffd.xyz"), "r") as ref, open(self.temp_ffd, "r") as actual:
            ref_lines = list(ref)
            actual_lines = list(actual)
            self.assertEqual(len(ref_lines), len(actual_lines))
            for i in range(len(ref_lines)):
                self.assertEqual(ref_lines[i], actual_lines[i])

    def test_writeP3D(self):
        self.foil.writeCoords(self.temp_p3d, spline_coords=True, file_format="plot3d")
        self.temp_p3d += ".xyz"
        with open(os.path.join(baseDir, "ref/rae2822_p3d.xyz"), "r") as ref, open(self.temp_p3d, "r") as actual:
            ref_lines = list(ref)
            actual_lines = list(actual)
            self.assertEqual(len(ref_lines), len(actual_lines))
            for i in range(len(ref_lines)):
                self.assertEqual(ref_lines[i], actual_lines[i])

    def test_writeDat(self):
        self.foil.writeCoords(self.temp_dat, spline_coords=True, file_format="dat")
        self.temp_dat += ".dat"
        with open(os.path.join(baseDir, "ref/rae2822_dat.dat"), "r") as ref, open(self.temp_dat, "r") as actual:
            ref_lines = list(ref)
            actual_lines = list(actual)
            self.assertEqual(len(ref_lines), len(actual_lines))
            for i in range(len(ref_lines)):
                self.assertEqual(ref_lines[i], actual_lines[i])

    def tearDown(self):
        for file_ in [self.temp_ffd, self.temp_p3d, self.temp_dat]:
            if os.path.isfile(file_):
                os.remove(file_)


if __name__ == "__main__":
    unittest.main()
