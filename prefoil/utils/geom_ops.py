"""

..      geom_ops
        --------
    Contains functions for modifying and creating airfoil geometries


"""

import numpy as np
from .. import sampling
from . import Error


def generateNACA(code, nPts, spacingFunc=sampling.cosine, func_args=None):
    """
    This function generates airfoil coordinates from the analytical definition of the NACA airfoils. It currently only supports 4 digit series airfoils.

    Parameters
    ----------
    code : str
        The 4 digit code, this is expected to be a length four string
    nPts : int
        The number of points to sample from the defintion of the NACA airfoil, half will be sampled on the top and half on the bottom
    spacingFunc : callable
        The spacing function to use for determining the sampling point locations of the x coordinates of the camber line
    func_args : dict
        Arguments to pass to the sampling function when it is called

    Returns
    -------
    af : Ndarray [N,2]
        Coordinates that were sampled from the NACA 4-digit code
    """

    if len(code) != 4:
        raise Error("Expected a NACA 4 digit code, but got %.d digits." % len(code))

    if not code.isdigit():
        raise Error("The NACA code provided was not made up of only digits.")

    if not func_args:
        func_args = {}

    camber_x = spacingFunc(0.0, 1.0, nPts // 2, **func_args)
    camber_y = np.zeros_like(camber_x)
    upper_x = np.zeros((nPts // 2, 1))
    lower_x = np.zeros_like(upper_x)
    upper_y = np.zeros_like(upper_x)
    lower_y = np.zeros_like(upper_x)

    m = int(code[0]) * 0.01
    p = int(code[1]) * 0.1
    t = int(code[2:]) * 0.01

    for i in range(len(camber_x)):
        if camber_x[i] < p:
            camber_y[i] = m / p**2 * (2 * p * camber_x[i] - camber_x[i] ** 2)
        else:
            camber_y[i] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * camber_x[i] - camber_x[i] ** 2)

    for i in range(len(camber_x)):
        thick_y = (
            t
            / 0.2
            * (
                0.2969 * np.sqrt(camber_x[i])
                - 0.126 * camber_x[i]
                - 0.3516 * camber_x[i] ** 2
                + 0.2843 * camber_x[i] ** 3
                - 0.1015 * camber_x[i] ** 4
            )
        )
        if camber_x[i] < p:
            theta = np.arctan(m / p**2 * (2 * p - 2 * camber_x[i]))
        else:
            theta = np.arctan(m / (1 - p) ** 2 * (2 * p - 2 * camber_x[i]))
        upper_x[i] = camber_x[i] - thick_y * np.sin(theta)
        lower_x[i] = camber_x[i] + thick_y * np.sin(theta)
        upper_y[i] = camber_y[i] + thick_y * np.cos(theta)
        lower_y[i] = camber_y[i] - thick_y * np.cos(theta)

    coords = np.hstack(
        (np.concatenate((np.flip(upper_x[1:]), lower_x)), np.concatenate((np.flip(upper_y[1:]), lower_y)))
    )

    return coords


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
    exc : Ndarray
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
        The angle in radians to rotate the coordinates

    origin : Ndarray [2]
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
