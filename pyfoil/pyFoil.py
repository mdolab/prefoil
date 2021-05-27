"""
pyFoil
--------

Contains a class for creating, modifying and exporting airfoils.


Questions:
- Modes?!? Should we provide any functionality for that?
- Do we want twist in deg or rad?
"""

import numpy as np
import pyspline as pySpline
from scipy.optimize import brentq, newton, bisect
from pyfoil import sampling
import pygeo


class Error(Exception):
    """
    Format the error message in a box to make it clear this
    was a explicitly raised exception.
    """

    def __init__(self, message):
        msg = "\n+" + "-" * 78 + "+" + "\n" + "| pyFoil Error: "
        i = 17
        for word in message.split():
            if len(word) + i + 1 > 78:  # Finish line and start new one
                msg += " " * (78 - i) + "|\n| " + word + " "
                i = 1 + len(word) + 1
            else:
                msg += word + " "
                i += len(word) + 1
        msg += " " * (78 - i) + "|\n" + "+" + "-" * 78 + "+" + "\n"
        print(msg)
        Exception.__init__()


def readCoordFile(filename, headerlines=0):
    """ Load the airfoil file"""
    f = open(filename, "r")
    line = f.readline()  # Read (and ignore) the first line
    r = []
    try:
        r.append([float(s) for s in line.split()])
    except:  # noqa
        r = []

    while True:
        line = f.readline()
        if not line:
            break  # end of file
        if line.isspace():
            break  # blank line
        r.append([float(s) for s in line.split()])

    X = np.array(r)
    return X


def _cleanup_pts(X):
    """
    DO NOT USE THIS, IT CURRENTLY DOES NOT WORK
    For now this just removes points which are too close together. In the future we may need to add further
    functionalities. This is just a generic cleanup tool which is called as part of preprocessing.
    """
    uniquePts, link = pygeo.geo_utils.pointReduce(X, nodeTol=1e-12)
    nUnique = len(uniquePts)

    # Create the mask for the unique data:
    mask = np.zeros(nUnique, "intc")
    for i in range(len(link)):
        mask[link[i]] = i

    # De-duplicate the data
    data = X[mask, :]
    return data


def _genNACACoords(name):
    pass


def _cleanup_TE(X, tol):
    TE = np.mean(X[[-1, 0], :], axis=0)
    return X, TE


def _writePlot3D(filename, x, y):
    filename += ".xyz"

    with open(filename, "w") as f:
        f.write("1\n")
        f.write("%d %d %d\n" % (len(x), 2, 1))
        for iDim in range(3):
            for j in range(2):
                for i in range(len(x)):
                    if iDim == 0:
                        f.write("%g\n" % x[i])
                    elif iDim == 1:
                        f.write("%g\n" % y[i])
                    else:
                        f.write("%g\n" % (float(j)))


def _writeDat(filename, x, y):
    filename += ".dat"

    with open(filename, "w") as f:
        for i in range(0, len(x)):
            f.write(str(round(x[i], 12)) + "\t\t" + str(round(y[i], 12)) + "\n")


def writeFFD(FFDbox, filename):
    """
    This function writes out an FFD in plot3D format from an FFDbox.

    Parameters
    ----------
    FFDBox : Ndarray [N,2,2,3]
        FFD Box to write out

    filename : str
        filename to write out, not including the '.xyz' ending

    """

    nffd = FFDbox.shape[0]

    # Write to file
    with open(filename + ".xyz", "w") as f:
        f.write("1\n")
        f.write(str(nffd) + " 2 2\n")
        for ell in range(3):
            for k in range(2):
                for j in range(2):
                    for i in range(nffd):
                        f.write("%.15f " % (FFDbox[i, j, k, ell]))
                    f.write("\n")


def _translateCoords(X, dX):
    """shifts the input coordinates by dx and dy"""
    return X + dX


def _rotateCoords(X, angle, origin):
    """retruns the coordinates rotated about the specified origin by angle (in deg)"""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))
    shifted_X = X - origin
    shifted_rotated_X = np.dot(shifted_X, R.T)
    rotated_X = shifted_rotated_X + origin
    return rotated_X


def _scaleCoords(X, scale, origin):
    """scales the coordinates in both dimension by the scaling factor"""
    shifted_X = X - origin
    shifted_scaled_X = shifted_X * scale
    scaled_X = shifted_scaled_X + origin
    return scaled_X


def checkCellRatio(X, ratio_tol=1.2):
    X_diff = X[1:, :] - X[:-1, :]
    cell_size = np.sqrt(X_diff[:, 0] ** 2 + X_diff[:, 1] ** 2)
    crit_cell_size = np.flatnonzero(cell_size < 1e-10)
    for i in crit_cell_size:
        print("critical I", i)
    cell_ratio = cell_size[1:] / cell_size[:-1]
    exc = np.flatnonzero(cell_ratio > ratio_tol)

    if exc.size > 0:
        print("WARNING: There are ", exc.size, " elements which exceed " "suggested cell ratio: ", exc)

    max_cell_ratio = np.max(cell_ratio, 0)
    avg_cell_ratio = np.average(cell_ratio, 0)
    print("Max cell ratio: ", max_cell_ratio)
    print("Average cell ratio", avg_cell_ratio)

    return cell_ratio, max_cell_ratio, avg_cell_ratio, exc


class Airfoil(object):
    """
    A class for manipulating airfoil geometry.

    Create an instance of an airfoil. There are two ways of instantiating
    this object: by passing in a set of points, or by reading in a coordinate
    file. The points must satisfy the following requirements:
        - Ordered such that they form a continuous airfoil surface
        - First and last points correspond to trailing edge

    It is not necessary for the points to be in a counter-clockwise ordering. If
    they are not ordered counter-clockwise, the order will be reversed so that
    all functions can be written to expect the spline to begin at the upper
    surface of the trailing edge and end at the lower surface of the trailing
    edge.

    Parameters
    ----------
    coords : ndarray[N,3]
        Full array of airfoil coordinates

    some additional option:

    k : int
        Order of the spline
    nCtl : int
        Number of control points
    name : str
        The name of the airfoil.


    Examples
    --------
    The general sequence of operations for using pyfoil is as follows::
      >>> from pygeo import *
    """

    def __init__(self, coords, spline_order=3, normalize=False):

        self.spline_order = spline_order

        # Initialize geometric information
        self.recompute(coords)

        if normalize:
            self.normalizeChord()

    def recompute(self, coords):
        self.spline = pySpline.Curve(X=coords, k=self.spline_order)
        self.reorder()

        self.TE = self.getTE()
        self.LE, self.s_LE = self.getLE()
        self.chord = self.getChord()
        self.twist = self.getTwist()
        self.closedCurve = (self.spline.getValue(0) == self.spline.getValue(1)).all()

    def reorder(self):
        """
        This function serves two purposes. First, it makes sure the points are oriented in counter-clockwise
        direction. Second, it makes sure the points start at the TE.
        """

        # Check to make sure spline ends at TE (For now assume this is True)

        # Make sure oriented in counter-clockwise direction.
        coords = self.spline.X
        N = coords.shape[0]

        orientation = 0
        for i in range(1, N - 1):
            v = coords[i + 1] - coords[i]
            r = coords[i + 1] - coords[i - 1]
            s = (coords[i, 0] * r[0] + coords[i, 1] * r[1]) / np.linalg.norm(r)
            n = coords[i] - r * s
            if np.linalg.norm(n) != 0:
                n = n / np.linalg.norm(n)
            orientation += n[0] * v[1] - n[1] * v[0]

        if orientation < 0:
            # Flipping orientation to counter-clockwise
            self.recompute(self.spline.X[::-1, :])

    ## Geometry Information

    def getTE(self):
        TE = (self.spline.getValue(0) + self.spline.getValue(1)) / 2
        return TE

    def getLE(self):
        """
        Calculates the leading edge point on the spline, which is defined as the point furthest away from the TE. The spline is assumed to start at the TE. The routine uses a root-finding algorithm to compute the LE.
        Let the TE be at point :math:`x_0, y_0`, then the Euclidean distance between the TE and any point on the airfoil spline is :math:`\ell(s) = \sqrt{\Delta x^2 + \Delta y^2}`, where :math:`\Delta x = x(s)-x_0` and :math:`\Delta y = y(s)-y_0`. We know near the LE, this quantity is concave. Therefore, to find its maximum, we differentiate and use a root-finding algorithm on its derivative.
        :math:`\\frac{\mathrm{d}\ell}{\mathrm{d}s} = \\frac{\Delta x\\frac{\mathrm{d}x}{\mathrm{d}s} + \Delta y\\frac{\mathrm{d}y}{\mathrm{d}s}}{\ell}`

        The function dellds computes the quantity :math:`\Delta x\\frac{\mathrm{d}x}{\mathrm{d}s} + \Delta y\\frac{\mathrm{d}y}{\mathrm{d}s}` which is then used by brentq to find its root, with an initial bracket at [0.3, 0.7].

        TODO
        Use a Newton solver, employing 2nd derivative information and use 0.5 as the initial guess.
        """

        def dellds(s, spline, TE):
            pt = spline.getValue(s)
            deriv = spline.getDerivative(s)
            dx = pt[0] - TE[0]
            dy = pt[1] - TE[1]
            return dx * deriv[0] + dy * deriv[1]

        s_LE = brentq(dellds, 0.3, 0.7, args=(self.spline, self.TE))
        LE = self.spline.getValue(s_LE)

        return LE, s_LE

    def getTwist(self):
        chord_vec = self.TE - self.LE
        twist = np.arctan2(chord_vec[1], chord_vec[0]) * 180 / np.pi
        # twist = np.arccos(normalized_chord.dot(np.array([1., 0.]))) * np.sign(normalized_chord[1])
        return twist

    def getChord(self):
        chord = np.linalg.norm(self.TE - self.LE)
        return chord

    def getPts(self):
        """alias for returning the points that make the airfoil spline"""
        return self.spline.X

    def findPt(self, position, axis=0, s_0=0):
        """finds that point at the intersection of the plane defined by the axis and the postion
        and the airfoil curve"""

        def err(s):
            return self.spline(s)[axis] - position

        def err_deriv(s):
            return self.spline.getDerivative(s)[axis]

        s_x = newton(err, s_0, fprime=err_deriv)

        return self.spline.getValue(s_x), s_x

    def getTEThickness(self):
        top = self.spline.getValue(0)
        bottom = self.spline.getValue(1)
        TE_thickness = np.array([top[0] + bottom[0], top[1] - bottom[1]]) / 2
        return TE_thickness

    def getLERadius(self):
        """
        Computes the leading edge radius of the airfoil. Note that this is heavily dependent on the initialization points, as well as the spline order/smoothing.
        """
        # if self.s_LE is None:
        #     self.getLE()

        first = self.spline.getDerivative(self.s_LE)
        second = self.spline.getSecondDerivative(self.s_LE)
        LE_rad = np.linalg.norm(first) ** 3 / np.linalg.norm(first[0] * second[1] - first[1] * second[0])
        return LE_rad

    def getCTDistribution(self):
        """
        Return the coordinates of the camber points, as well as the thicknesses (this is with british convention).
        """
        self._splitAirfoil()

        num_chord_pts = 100

        # Compute the chord
        chord_pts = np.vstack([self.LE, self.TE])
        chord = pySpline.line(chord_pts)

        cos_sampling = np.linspace(0, 1, num_chord_pts + 1, endpoint=False)[1:]
        # cos_sampling = smp.conical(num_chord_pts+2,coeff=1)[1:-1]

        chord_pts = chord.getValue(cos_sampling)
        camber_pts = np.zeros((num_chord_pts, 2))
        thickness_pts = np.zeros((num_chord_pts, 2))
        for j in range(chord_pts.shape[0]):
            direction = np.array([np.cos(np.pi / 2 - self.twist), np.sin(np.pi / 2 - self.twist)])
            direction = direction / np.linalg.norm(direction)
            top = chord_pts[j, :] + 0.5 * self.chord * direction
            bottom = chord_pts[j, :] - 0.5 * self.chord * direction
            temp = np.vstack((top, bottom))
            normal = pySpline.line(temp)
            s_top, t_top, D = self.top.projectCurve(normal, nIter=5000, eps=1e-16)
            s_bottom, t_bottom, D = self.bottom.projectCurve(normal, nIter=5000, eps=1e-16)
            intersect_top = self.top.getValue(s_top)
            intersect_bottom = self.bottom.getValue(s_bottom)

            # plt.plot(temp[:,0],temp[:,1],'-og')
            # plt.plot(intersect_top[0],intersect_top[1],'or')
            # plt.plot(intersect_bottom[0],intersect_bottom[1],'ob')

            camber_pts[j, :] = (intersect_top + intersect_bottom) / 2
            thickness_pts[j, 0] = (intersect_top[0] + intersect_bottom[0]) / 2
            thickness_pts[j, 1] = (intersect_top[1] - intersect_bottom[1]) / 2
        # plt.plot(camber_pts[:,0],camber_pts[:,1],'ok')

        self.camber_pts = np.vstack((self.LE, camber_pts, self.TE))  # Add TE and LE to the camber points.
        self.getTEThickness()
        self.thickness_pts = np.vstack((np.array((self.LE[0], 0)), thickness_pts, self.TE_thickness))

        return self.camber_pts, self.thickness_pts

    def getTEAngle(self):
        """
        Computes the trailing edge angle of the airfoil. We assume here that the spline goes from top to bottom, and that s=0 and s=1 corresponds to the
        top and bottom trailing edge points. Whether or not the airfoil is closed is irrelevant.
        """
        top = self.spline.getDerivative(0)
        top = top / np.linalg.norm(top)
        bottom = self.spline.getDerivative(1)
        bottom = bottom / np.linalg.norm(bottom)
        # print(np.dot(top,bottom))
        self.TE_angle = np.pi - np.arccos(np.dot(top, bottom))
        return np.rad2deg(self.TE_angle)

    # TODO write
    def getMaxThickness(self, method):
        """
        method : str
            Can be one of 'british', 'american', or 'projected'
        """
        pass

    def getMaxCamber(self):
        pass

    def _getClosest(self, coords, x):
        """
        Gets the closest y value on the upper and lower point to an x value

        Parameters
        ----------
        coords : Ndarray [N,2]
            coordinates defining the airfoil

        x : float
            The x value to find the closest point for
        """
        # TODO should this be modified to interpolate points using findPts?

        top = coords[: len(coords + 1) // 2 + 1, :]
        bottom = coords[len(coords + 1) // 2 :, :]

        x_top = np.ones(len(top))
        for i in range(len(top)):
            x_top[i] = abs(top[i, 0] - x)
        yu = top[np.argmin(x_top), 1]
        x_bottom = np.ones(len(bottom))
        for i in range(len(bottom)):
            x_bottom[i] = abs(bottom[i, 0] - x)
        yl = bottom[np.argmin(x_bottom), 1]

        return yu, yl

    def isReflex(self):
        """
        An airfoil is reflex if the derivative of the camber line at the trailing edge is positive.
        #TODO this has not been tested
        """
        if self.camber is None:
            self.getCamber()

        if self.camber.getDerivative(1)[1] > 0:
            return True
        else:
            return False

    def isSymmetric(self, tol=1e-6):
        # test camber and thickness dist
        pass

    # ==============================================================================
    # Geometry Modification
    # ==============================================================================

    def rotate(self, angle, origin=np.zeros(2)):
        new_coords = _rotateCoords(self.spline.X, np.deg2rad(angle), origin)

        # reset initialize with the new set of coordinates
        self.__init__(new_coords, spline_order=self.spline.k)
        # self.update(new_coords, spline_order=self.spline.k)

    def derotate(self, origin=np.zeros(2)):
        self.rotate(-1.0 * self.twist, origin=origin)

    def scale(self, factor, origin=np.zeros(2)):
        new_coords = _scaleCoords(self.spline.X, factor, origin)
        self.__init__(new_coords, spline_order=self.spline.k)

    def normalizeChord(self, origin=np.zeros(2)):
        if self.spline is None:
            self.recompute()
        elif self.chord == 1:
            return
        self.scale(1.0 / self.chord, origin=origin)

    def translate(self, delta):
        sample_pts = self._getDefaultSampling()
        self.X = _translateCoords(sample_pts, delta)
        self.recompute()
        if self.LE is not None:
            self.LE += delta

    def center(self):
        if self.spline is None:
            self.recompute()
        if self.LE is None:
            self.getChord()
        elif np.all(self.LE == np.zeros(2)):
            return
        self.translate(-1.0 * self.LE)

    def splitAirfoil(self):
        # if self.s_LE is None:
        # self.getLE()
        top, bottom = self.spline.splitCurve(self.s_LE)
        return top, bottom

    def normalizeAirfoil(self, derotate=True, normalize=True, center=True):
        if derotate or normalize or center:
            origin = np.zeros(2)
            sample_pts = self.spline.X

            # Order of operation here is important, even though all three operations are linear, because
            # we rotate about the origin for simplicity.
            if center:
                delta = -1.0 * self.LE
                sample_pts = _translateCoords(sample_pts, delta)
            if derotate:
                angle = -1.0 * self.twist
                sample_pts = _rotateCoords(sample_pts, angle, origin)
            if normalize:
                factor = 1.0 / self.chord
                sample_pts = _scaleCoords(sample_pts, factor, origin)

            self.recompute(sample_pts)

    def makeBluntTE(self, start=0.01, end=None):
        """
        This cuts the upper surface at s=start and the lower surface at s=end
        and creates a blunt trailing edge between the two cut points. If end
        is not provided, then the cut is made on the upper surface and projected
        down to the lower surface along the y-axis.
        """
        if end is None:
            xstart = self.spline.getValue(start)
            # Make the trailing edge parallel with y-axis

            def findEnd(s):
                xend = self.spline.getValue(s)
                return xend[0] - xstart[0]

            end = bisect(findEnd, 0.9, 1.0)

        newCurve = self.spline.windowCurve(start, end)
        coords = newCurve.getValue(self.spline.gpts)
        self.recompute(coords)

    def sharpenTE(self):
        pass

    def roundTE(self, xCut=0.98, k=4, nPts=20):
        """this method creates a smooth round trailing edge **from a blunt one** using a spline

        Parameters
        ----------
        xCut : float
            x location of the cut **as a percentage of the chord**
        K: int (3 or 4)
            order of the spline used to make the rounded trailing edge of the airfoil.
        nPts : int
            Number of trailing edge points to add to the airfoil spline

        """
        # convert the xCut loctation from a percentage to an abs value
        xCut = self.LE[0] + xCut * (self.TE[0] - self.LE[0])
        dx = self.TE[0] - xCut

        # create the knot vector for the spline
        t = [0] * k + [0.5] + [1] * k

        # create the vector of control points for the spline
        coeff = np.zeros((k + 1, 2))

        for ii in [0, -1]:
            coeff[ii], s = self.findPt(xCut, s_0=np.abs(ii))
            # coeff[-1], s_lower  = self.findPt(xCut, s_0=1)
            dX_ds = self.spline.getDerivative(s)
            dy_dx = dX_ds[0] ** -1 * dX_ds[1]

            # the indexing here is a bit confusing.ii = 0 -> coeff[1] and ii = -1 -> coef[-2]
            coeff[3 * ii + 1] = np.array([self.TE[0], coeff[ii, 1] + dy_dx * dx])

        if k == 4:
            coeff[2] = self.TE

        ## make the TE curve
        te_curve = pySpline.Curve(t=t, k=k, coef=coeff)

        # ----- combine the TE curve with the spline curve -----
        upper_curve, lower_curve = te_curve.splitCurve(0.5)
        upper_pts = upper_curve.getValue(np.linspace(0, 1, nPts))
        lower_pts = lower_curve.getValue(np.linspace(0, 1, nPts))
        # remove the replaced pts
        mask = []
        for ii in range(self.spline.X.shape[0]):
            if xCut > self.spline.X[ii, 0]:
                mask.append(ii)

        coords = np.vstack((upper_pts[::-1], self.spline.X[mask], lower_pts[::-1]))

        # ---- recompute with new TE ---
        self.recompute(coords)

    def removeTE(self, tol=0.3, xtol=0.9):
        """  """
        coords = self.getPts()
        chord_vec = self.TE - self.LE
        unit_chord_vec = chord_vec / np.linalg.norm(chord_vec)

        airfoil_mask = []
        TE_mask = []
        for ii in range(coords.shape[0] - 1):  # loop over each element

            if coords[ii, 0] >= (self.LE + chord_vec * xtol)[0]:
                delta = coords[ii + 1] - coords[ii]
                unit_delta = delta / np.linalg.norm(delta)

                if np.abs(np.dot(unit_chord_vec, unit_delta)) < tol:
                    TE_mask += [ii, ii + 1]
                else:
                    airfoil_mask += [ii, ii + 1]
            else:
                airfoil_mask += [ii, ii + 1]

        # list(set()) removes the duplicate pts
        self.recompute(coords[list(set(airfoil_mask))])

        return coords[list(set(TE_mask))]

    ## Sampling
    def getSampledPts(self, nPts, spacingFunc=sampling.polynomial, func_args={}, nTEPts=0):
        """
        This function defines the point sampling along airfoil surface. The
        coordinates are given as a closed curve (i.e. the first and last point
        are the same, regardless of whether the spline is closed or open).
        An example dictionary is reported below:

        >>> sample_dict = {'distribution' : 'conical',
        >>>        'coeff' : 1,
        >>>        'npts' : 50,
        >>>        'bad_edge': False}

        The point distribution currently implemented are:
            - *Cosine*:
            - *Conical*:
            - *Parabolic*:
            - *Polynomial*:

        :param upper: dictionary
                Upper surface sampling dictionary
        :param lower: dictionary
                Lower surface sampling dictionary
        :param npts_TE: float
                Number of points along the **blunt** trailing edge
        :return: Coordinates array, anticlockwise, from trailing edge
        """
        s = sampling.joinedSpacing(nPts, spacingFunc=spacingFunc, func_args=func_args)
        coords = self.spline.getValue(s)

        if nTEPts:
            coords_TE = np.zeros((nTEPts + 2, coords.shape[1]))
            for idim in range(coords.shape[1]):
                val1 = self.spline.getValue(1)[idim]
                val2 = self.spline.getValue(0)[idim]
                coords_TE[:, idim] = np.linspace(val1, val2, nTEPts + 2)
            coords = np.vstack((coords, coords_TE[1:-1]))

        if not self.closedCurve:
            coords = np.vstack((coords, coords[0]))

        # TODO
        # - reintagrate cell check

        # if cell_check is True:
        #     checkCellRatio(coords)
        # self.sampled_X = coords

        # # To be updated later on if new point add/remove operations are included
        # # x.size-1 because of the last point added for "closure"
        # if single_distr is True and x.size-1 != points_init:
        #     print('WARNING: The number of sampling points has been changed \n'
        #             '\t\tCurrent points number: %i' % (x.size))

        return coords

    def _buildFFD(self, nffd, fitted, xmargin, ymarginu, ymarginl, xslice, coords):
        """
        The function that actually builds the FFD Box from all of the given parameters

        Parameters
        ----------
        nffd : int
            number of FFD points along the chord

        fitted : bool
            flag to pick between a fitted FFD (True) and box FFD (False)

        xmargin : float
            The closest distance of the FFD box to the tip and aft of the airfoil

        ymarginu : float
            When a box ffd is generated this specifies the top of the box's y values as
            the maximum y value in the airfoil coordinates plus this margin.
            When a fitted ffd is generated this is the margin between the FFD point at
            an xslice location and the upper surface of the airfoil at this location

        ymarginl : float
            When a box ffd is generated this specifies the bottom of the box's y values as
            the minimum y value in the airfoil coordinates minus this margin.
            When a fitted ffd is generated this is the margin between the FFD point at
            an xslice location and the lower surface of the airfoil at this location

        xslice : Ndarray [N,2]
            User specified xslice locations. If this is chosen nffd is ignored

        coords : Ndarray [N,2]
            the coordinates to use for defining the airfoil, if the user does not
            want the original coordinates for the airfoil used. This shouldn't be
            used unless the user wants fine tuned control over the FFD creation,
            It should be sufficient to ignore.

        """

        if coords is None:
            coords = self.getPts()

        if xslice is None:
            xslice = np.zeros(nffd)
            for i in range(nffd):
                xtemp = i * 1.0 / (nffd - 1.0)
                xslice[i] = min(coords[:, 0]) - 1.0 * xmargin + (max(coords[:, 0]) + 2.0 * xmargin) * xtemp
        else:
            nffd = len(xslice)

        FFDbox = np.zeros((nffd, 2, 2, 3))

        if fitted:
            ylower = np.zeros(nffd)
            yupper = np.zeros(nffd)
            for i in range(nffd):
                ymargin = ymarginu + (ymarginl - ymarginu) * xslice[i]
                yu, yl = self._getClosest(coords, xslice[i])
                yupper[i] = yu + ymargin
                ylower[i] = yl - ymargin
        else:
            yupper = np.ones(nffd) * (max(coords[:, 1]) + ymarginu)
            ylower = np.ones(nffd) * (min(coords[:, 1]) - ymarginl)

        # X
        FFDbox[:, 0, 0, 0] = xslice[:].copy()
        FFDbox[:, 1, 0, 0] = xslice[:].copy()
        # Y
        # lower
        FFDbox[:, 0, 0, 1] = ylower[:].copy()
        # upper
        FFDbox[:, 1, 0, 1] = yupper[:].copy()
        # copy
        FFDbox[:, :, 1, :] = FFDbox[:, :, 0, :].copy()
        # Z
        FFDbox[:, :, 0, 2] = 0.0
        # Z
        FFDbox[:, :, 1, 2] = 1.0

        return FFDbox

    ## Output
    def writeCoords(self, coords, filename, format="plot3d"):
        """
        We have to discuss which types of printfiles we want to get and how
        to handle them (does this class print the last "sampled" x,y or do
        we want more options?)
        """

        if format == "plot3d":
            _writePlot3D(filename, coords[:, 0], coords[:, 1])
        elif format == "dat":
            _writeDat(filename, coords[:, 0], coords[:, 1])
        else:
            raise Error(format + " is not a supported output format!")

    def generateFFD(
        self, nffd, filename, fitted=True, xmargin=0.001, ymarginu=0.02, ymarginl=0.02, xslice=None, coords=None
    ):
        """
        Generates an FFD from the airfoil and writes it out to file

        nffd : int
            the number of chordwise points in the FFD

        filename : str
            filename to write out, not including the '.xyz' ending

        fitted : bool
            flag to pick between a fitted FFD (True) and box FFD (False)

        xmargin : float
            The closest distance of the FFD box to the tip and aft of the airfoil

        ymarginu : float
            When a box ffd is generated this specifies the top of the box's y values as
            the maximum y value in the airfoil coordinates plus this margin.
            When a fitted ffd is generated this is the margin between the FFD point at
            an xslice location and the upper surface of the airfoil at this location

        ymarginl : float
            When a box ffd is generated this specifies the bottom of the box's y values as
            the minimum y value in the airfoil coordinates minus this margin.
            When a fitted ffd is generated this is the margin between the FFD point at
            an xslice location and the lower surface of the airfoil at this location

        xslice : Ndarray [N,2]
            User specified xslice locations. If this is chosen nffd is ignored

        coords : Ndarray [N,2]
            the coordinates to use for defining the airfoil, if the user does not
            want the original coordinates for the airfoil used. This shouldn't be
            used unless the user wants fine tuned control over the FFD creation,
            It should be sufficient to ignore.
        """

        FFDbox = self._buildFFD(nffd, fitted, xmargin, ymarginu, ymarginl, xslice, coords)

        writeFFD(FFDbox, filename)

    ## Utils
    # maybe remove and put into a separate location?
    def plot(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        # pts = self._getDefaultSampling(npts=1000)
        plt.plot(self.spline.X[:, 0], self.spline.X[:, 1], "-r")
        plt.axis("equal")
        # if self.sampled_X is not None:
        plt.plot(self.spline.X[:, 0], self.spline.X[:, 1], "o")

        # TODO
        # if self.camber_pts is not None:
        #     fig2 = plt.figure()
        #     plt.plot(self.camber_pts[:,0],self.camber_pts[:,1],'-og',label='camber')
        #     plt.plot(self.thickness_pts[:,0],self.thickness_pts[:,1],'-ob',label='thickness')
        #     plt.legend(loc='best')
        #     plt.title(self.name)
        return fig
