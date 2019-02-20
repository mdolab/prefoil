import numpy as np
from scipy import optimize


# for linear use linspace

def cosine(start, end, n, m=np.pi):
    s = conical(start, end, n, m, coeff=1)
    return s


def conical(start, end, n, m=np.pi, coeff=1, bad_edge=False):
    '''
    Here I make a sneaky hack to remove the second and second to last points
    of the ndarray. This is necessary to avoid bad mesh elements at small
    leading and trailing edges
    '''

    b = coeff

    if bad_edge is True:
        x = np.linspace(m, 0, n + 2)
        x = np.delete(np.delete(x, 1), -2)
    else:
        x = np.linspace(m, 0, n)
    if b >= 1:
        s = (1 + 1 / np.sqrt(np.cos(x)**2 + np.sin(x)**2 / b**2) * np.cos(x)) * 0.5
    else:
        cos = np.cos(x)
        s = ((cos + 1) / 2 - x[::-1] / np.pi) * b + x[::-1] / np.pi

    return s * (end - start) + start


def polynomial(start, end, n, m=np.pi, order=5):
    """
        similar to cosSacing but instead of a unit circle, a func of the form 1 - x^order is used.
        This does a better job on not overly clustering points at the edges.

                ---------------|---------------
              -/            -/ | \-            \-    order 4
            -/            -/   |   \-            \-
          -/            -/     |     \-            \-
        -/            -/       |       \-            \-
        |           -/         |         \-           |
        |         -/           | order 1   \-         |
        |       -/             |             \-       |
        |     -/               |               \-     |
        |   -/                 |                 \-   |
        | -/                   |                   \- |
        |/                     |                     \|
        ------------------------------------------------
    """

    def poly(x):
        return np.abs(x)**order + np.tan(ang) * x - 1

    angles = np.linspace(m, 0, n)

    s = np.array([])
    for ang in angles:
        s = np.append(s, optimize.fsolve(poly, np.cos(ang))[0])

    return (s / 2 + 0.5) * (end - start) + start


def joinedSpacing(n, spacingFunc=polynomial, func_args={}, s_LE=0.5, closedCurve=True):
    """
    function that returns two point distributions joined at s_LE

                        s1                            s2
    || |  |   |    |     |     |    |   |  | |||| |  |   |    |     |     |    |   |  | ||
                                                /\
                                                s_LE

    Note that one point is added when sampling due to the removal of "double"
    elements when returning the point array
    """
    s1 = spacingFunc(0., s_LE, int(n * s_LE) + 1, **func_args)
    s2 = spacingFunc(s_LE, 1., int(n - n * s_LE) + 1, **func_args)

    # if not closedCurve:
    #     s2 = s2[:-1]

    # combine the two distributions
    s = np.append(s1[:], s2[1:])

    return s
