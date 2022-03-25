"""

..		geom_ops
		--------
	Contains functions for modifying and creating airfoil geometries


"""

import numpy as np


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
