"""

..      io_utils
        --------

    Contains helper functions for file and user I/O

"""

import numpy as np


class Error(Exception):
    """
    Formats error messages to make it clear that it was explicitly raised by prefoil.
    """

    def __init__(self, message):
        msg = "\n+" + "-" * 78 + "+" + "\n" + "| preFoil Error: "
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
        super().__init__(msg)


def readCoordFile(filename, headerlines=0):
    """
    This function reads in a '.dat' style airfoil coordinate file,
    with each coordinate on a new line and each line containing an xy pair
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
    with open(filename, "r") as f:
        for _i in range(headerlines):
            f.readline()
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


def _writePlot3D(filename, x, y):
    """
    This function writes out a 2D airfoil surface in 3D (one element in z direction)

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
        for i in range(len(x)):
            f.write(str(round(x[i], 12)) + "\t\t" + str(round(y[i], 12)) + "\n")


def _writeFFD(FFDbox, filename):
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
