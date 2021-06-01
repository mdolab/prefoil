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
    """
    This function reads in a '.dat' style airfoil coordinate file.
    With each coordinate on a new line and each line containing an xy pair
    separate by whitespace

    Parameters
    ----------
    filename : str
        the file to read from

    headerlines : int
        the number of lines to skip at the beginning of the file to reach the coordinates

    Returns
    -------
    X : Ndarray [N,2]
        The coordinates read from the file
    """
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
    """
    This function writes out a airfoil 2D airfoil surface in 3D (one element in z direction)

    Parameters
    ----------
    filename : str
        filename to write out, not including the '.xyz' ending

    x : Ndarray [N]
        a list of all the x values of the coordinates

    y : Ndarray [N]
        a list of all the y values of the coordinates

    """
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
    """
    This function writes out coordinates in a space delimited list

    Parameters
    ----------
    filename : str
        filename to write out, not including the '.dat' ending

    x : Ndarray [N]
        a list of all the x values of the coordinates

    y : Ndarray [N]
        a list of all the y values of the coordinates

    """

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
    """
    Translates the input coordinates

    Parameters
    ----------
    X : Ndarray [N,2]
        The x/y coordinate pairs that are being translated

    dX : Ndarray [N,2]
        The dx/dy amount to translate in each direction

    Returns
    -------
    translated_X : Ndarray [N,2]
        The translated coordinates
    """
    return X + dX


def _rotateCoords(X, angle, origin):
    """
    Rotates coordinates about the specified origin by angle (in deg)

    Parameters
    ----------
    X : Ndarray [N,2]
        The x/y coordinate pairs that are being rotated

    angle : float
        The angle in degrees to rotate the coordinates

    origin : 2darray [2]
        The x/y coordinate pair specifying the rotation origin

    Returns
    -------
    rotated_X : Ndarray[N,2]
        The rotated coordinate pairs
    """
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))
    shifted_X = X - origin
    shifted_rotated_X = np.dot(shifted_X, R.T)
    rotated_X = shifted_rotated_X + origin
    return rotated_X


def _scaleCoords(X, scale, origin):
    """
    Scales coordinates in both dimensions by the scaling factor from a given origin

    Parameters
    ----------
    X : Ndarry [N,2]
        The x/y coordinate pairs that are being scaled

    scale : float
        The scaling factor

    origin : float
        The location about which scaling occurs (This point will not change)

    Returns
    -------
    scaled_X : Ndarray [N,2]
        the scaled coordinate values
    """
    shifted_X = X - origin
    shifted_scaled_X = shifted_X * scale
    scaled_X = shifted_scaled_X + origin
    return scaled_X


def checkCellRatio(X, ratio_tol=1.2):
    """
    Checks a set of coordinates for consecutive cell ratios that exceed a given tolerance

    Parameters
    ----------
    X : Ndarray [N,2]
        The set of coordinates being checked

    ratio_tol : float
        The maximum cell ratio that is allowed

    Returns
    -------
    cell_ratio : Ndarray [N]
        the cell ratios for each cell

    max_cell_ratio : float
        the maximum cell ratio

    avg_cell_ratio : float
        the average cell ratio

    exc : Ndarray [N]
        the cell indicies that exceed the ratio tolerance
    """
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


def _getClosestY(coords, x):
    """
    Gets the closest y value on the upper and lower point to an x value

    Parameters
    ----------
    coords : Ndarray [N,2]
        coordinates defining the airfoil

    x : float
        The x value to find the closest point for

    Returns
    -------
    yu : float
        The y value of the closest coordinate on the upper surface

    yl : float
        The y value of the closest coordinate on the lower surface
    """
    # TODO should this be modified to use the spline from the airfoil?

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

    spline_order : int
        Order of the spline

    normalize : bool
        True to normalize the chord of the airfoil

    Examples
    --------
    The general sequence of operations for using pyfoil is as follows::
      >>> from pyfoil.pyFoil import *
    """

    def __init__(self, coords, spline_order=3, normalize=False):

        self.spline_order = spline_order

        # Initialize geometric information
        self.recompute(coords)

        if normalize:
            self.normalizeChord()

    def recompute(self, coords):
        """
        Recomputes the underlying spline and relevant parameters from the given set of coordinate

        Parameters
        ----------
        coords : Ndarray [N,2]
            The coordinate pairs to compute the airfoil spline from

        """
        self.spline = pySpline.Curve(X=coords, k=self.spline_order)
        self.reorder()

        self.TE = self.getTE()
        self.LE, self.s_LE = self.getLE()
        self.chord = self.getChord()
        self.twist = self.getTwist()
        self.closedCurve = (self.spline.getValue(0) == self.spline.getValue(1)).all()

    def reorder(self):
        """
        This function orients the points counterclockwise and sets the start point to the TE
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
        """
        Calculates the trailing edge point on the spline

        Returns
        -------
        TE : 2darry [2]
            The coordinate of the trailing edge of the airfoil
        """

        TE = (self.spline.getValue(0) + self.spline.getValue(1)) / 2
        return TE

    def getLE(self):
        """
        Calculates the leading edge point on the spline, which is defined as the point furthest away from the TE. The spline is assumed to start at the TE. The routine uses a root-finding algorithm to compute the LE.
        Let the TE be at point :math:`x_0, y_0`, then the Euclidean distance between the TE and any point on the airfoil spline is :math:`\ell(s) = \sqrt{\Delta x^2 + \Delta y^2}`, where :math:`\Delta x = x(s)-x_0` and :math:`\Delta y = y(s)-y_0`. We know near the LE, this quantity is concave. Therefore, to find its maximum, we differentiate and use a root-finding algorithm on its derivative.
        :math:`\\frac{\mathrm{d}\ell}{\mathrm{d}s} = \\frac{\Delta x\\frac{\mathrm{d}x}{\mathrm{d}s} + \Delta y\\frac{\mathrm{d}y}{\mathrm{d}s}}{\ell}`

        The function dellds computes the quantity :math:`\Delta x\\frac{\mathrm{d}x}{\mathrm{d}s} + \Delta y\\frac{\mathrm{d}y}{\mathrm{d}s}` which is then used by brentq to find its root, with an initial bracket at [0.3, 0.7].

        Returns
        -------
        LE : 2darray [2]
            the coordinate of the leading edge

        s_LE : float
            the parametric position of the leading edge
        """

        # TODO Use a Newton solver, employing 2nd derivative information and use 0.5 as the initial guess.

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
        """
        Calculates the twist of the airfoil using the leading and trailing edge points

        Returns
        -------
        twist : float
            The twist in degrees of the airfoil

        """

        chord_vec = self.TE - self.LE
        twist = np.arctan2(chord_vec[1], chord_vec[0]) * 180 / np.pi
        # twist = np.arccos(normalized_chord.dot(np.array([1., 0.]))) * np.sign(normalized_chord[1])
        return twist

    def getChord(self):
        """
        Calculates the chord of the airfoil as the distance between the leading and trailing edges

        Returns
        -------
        chord : float
            The chord length
        """

        chord = np.linalg.norm(self.TE - self.LE)
        return chord

    def getPts(self):
        """
        alias for returning the points that make the airfoil spline

        Returns
        -------
        X : Ndarry [N, 2]
            the coordinates that define the airfoil spline
        """
        return self.spline.X

    def findPt(self, position, axis=0, s_0=0):
        """
        finds that point at the intersection of the plane defined by the axis and the postion
        and the airfoil curve

        Parameters
        ----------
        position : float
            the position of the plane on the given axis

        axis : int
            the axis the plane will intersect 0 for x and 1 for y

        s_0 : float
            an initial guess for the parameteric position of the solution

        Returns
        -------
        X : 2darray [2]
            The coordinate at the intersection

        s_x : float
            the parametric location of the intersection
        """

        def err(s):
            return self.spline(s)[axis] - position

        def err_deriv(s):
            return self.spline.getDerivative(s)[axis]

        s_x = newton(err, s_0, fprime=err_deriv)

        return self.spline.getValue(s_x), s_x

    def getTEThickness(self):
        """
        gets the trailing edge thickness for the airfoil

        Returns
        -------
        TE_thickness : float
            the trailing edge thickness
        """
        top = self.spline.getValue(0)
        bottom = self.spline.getValue(1)
        TE_thickness = np.array([top[0] + bottom[0], top[1] - bottom[1]]) / 2
        return TE_thickness

    def getLERadius(self):
        """
        Computes the leading edge radius of the airfoil. Note that this is heavily dependent on the initialization points, as well as the spline order/smoothing.

        Returns
        -------
        LE_rad : float
            The leading edge radius
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

        Returns
        -------
        camber_pts : Ndarray [N, 2]
            the locations of the camber points of the airfoil

        thickness_pts : Ndarray [N]
            the thickness of the airfoil at each camber point
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

        Returns
        -------
        TE_angle : float
            The angle of the trailing edge in degrees
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
        Parameters
        ----------
        method : str
            Can be one of 'british', 'american', or 'projected'

        Returns
        -------
        max_thickness : float
            the maximum thickness of the airfoil

        """
        pass

    def getMaxCamber(self):
        """
        Returns
        -------
        max_camber : float
            the maximum camber of the airfoil

        """
        pass

    def isReflex(self):
        """
        Determines if an airfoil is reflex by checking if the derivative of the camber line at the trailing edge is positive.

        Returns
        -------
        reflexive : bool
            True if reflexive
        """
        # TODO this has not been tested
        if self.camber is None:
            self.getCamber()

        if self.camber.getDerivative(1)[1] > 0:
            return True
        else:
            return False

    def isSymmetric(self, tol=1e-6):
        """
        Checks if an airfoil is symmetric

        Parameters
        ----------
        tol : float
            tolerance for camber line to still be consdiered symmetrical

        Returns
        -------
        symmetric : bool
            True if the airfoil is symmetric within the given tolerance
        """
        pass

    # ==============================================================================
    # Geometry Modification
    # ==============================================================================

    def rotate(self, angle, origin=np.zeros(2)):
        """
        rotates the airfoil about the specified origin

        Parameters
        ----------
        angle : float
            the angle to rotate the airfoil in degrees

        origin : 2darray [2]
            the point about which to rotate the airfoil
        """
        new_coords = _rotateCoords(self.spline.X, np.deg2rad(angle), origin)

        # reset initialize with the new set of coordinates
        self.__init__(new_coords, spline_order=self.spline.k)
        # self.update(new_coords, spline_order=self.spline.k)

    def derotate(self, origin=np.zeros(2)):
        """
        derotates the airfoil about the origin by the twist

        Parameters
        ----------
        origin : 2darray [2]
            the location about which to preform the rotation
        """
        self.rotate(-1.0 * self.twist, origin=origin)

    def scale(self, factor, origin=np.zeros(2)):
        """
        Scale the airfoil by factor about the origin

        Parameters
        ----------
        factor : float
            the scaling factor

        origin : 2darray [2]
            the coordinate about which to preform the scaling
        """

        new_coords = _scaleCoords(self.spline.X, factor, origin)
        self.__init__(new_coords, spline_order=self.spline.k)

    def normalizeChord(self, origin=np.zeros(2)):
        """
        Set the chord to 1 by scaling the airfoil about the given origin

        Parameters
        ----------
        origin : 2darray [2]
            the point about which to scale the airfoil
        """

        if self.spline is None:
            self.recompute()
        elif self.chord == 1:
            return
        self.scale(1.0 / self.chord, origin=origin)

    def translate(self, delta):
        """
        Translate the airfoil by the vector delta

        Parameters
        ----------
        delta : 2darray [2]
            the vector that defines the translation of the airfoil
        """

        sample_pts = self._getDefaultSampling()
        self.X = _translateCoords(sample_pts, delta)
        self.recompute()
        if self.LE is not None:
            self.LE += delta

    def center(self):
        """
        Move the airfoil so that the leading edge is at the origin
        """

        if self.spline is None:
            self.recompute()
        if self.LE is None:
            self.getChord()
        elif np.all(self.LE == np.zeros(2)):
            return
        self.translate(-1.0 * self.LE)

    def splitAirfoil(self):
        """
        Splits the airfoil into upper and lower surfaces

        Returns
        -------
        top : pySpline curve object
            A spline that defines the upper surface

        bottom : pySpline curve object
            A spline that defines the lower surface
        """

        # if self.s_LE is None:
        # self.getLE()
        top, bottom = self.spline.splitCurve(self.s_LE)
        return top, bottom

    def normalizeAirfoil(self, derotate=True, normalize=True, center=True):
        """
        Sets the twist to zero, the chord to one, and the leading edge location to the origin

        Parameters
        ----------
        derotate : bool
            True to set twist to zero

        normalize : bool
            True to set the chord length to one

        center : bool
            True to put the leading edge at the origin
        """

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
        This cuts the upper and lower surfaces to creates a blunt trailing edge between the two cut points.

        Parameters
        ----------
        start : float
            the parametric value at which to cut the upper surface

        end : float
            the parameteric value at which to cut the lower surface. If end is not provided,
            then the cut is made on the upper surface and projected down to the lower surface along the y-axis.
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
        """
        this method creates a smooth round trailing edge **from a blunt one** using a spline

        Parameters
        ----------
        xCut : float
            x location of the cut **as a percentage of the chord**
        k: int (3 or 4)
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
        >>>        'bad_edge': False}

        The point distribution currently implemented are:
            - *Cosine*:
            - *Conical*:
            - *Parabolic*:
            - *Polynomial*:

        Parameters
        ----------
        upper: dictionary
            Upper surface sampling dictionary
        lower: dictionary
            Lower surface sampling dictionary
        npts_TE: float
            Number of points along the **blunt** trailing edge

        Returns
        -------
        coords : Ndarray [N, 2]
            Coordinates array, anticlockwise, from trailing edge
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
                yu, yl = _getClosestY(coords, xslice[i])
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

        Parameters
        ----------
        coords : Ndarray [N,2]
            the coordinates to write out to a file

        filename : str
            the filename without extension to write to

        format : str
            the file format to write, can be `plot3d` or `dat`
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

        Parameters
        ----------
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
        """
        Plots the airfoil

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure with the plotted airfoil
        """

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
