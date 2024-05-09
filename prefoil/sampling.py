import numpy as np
from scipy import optimize


# for linear use linspace


def cosine(start, end, n, m=np.pi):
    r"""
    Sampling function based on cosine spacing. Check :meth:`prefoil.sampling.conical()` for more implementation information.

    Parameters
    ----------
    start : float
        the airfoil chord location to start sampling at

    end : float
        the airfoil chord location to stop sampling at

    n : int
        the number of points to sample

    m : float
        the maximum angle used for sampling the point distribution, starting from zero.
        This implicitly defines the "frequency" of the refinement, e.g. :math:`m=\pi` refinement at LE and TE, :math:`m=2\pi` refinement at LE, TE, and mid-chord, etc

    Returns
    -------
    s : Ndarray [N]
        The parametric spline locations that define the sampling
    """
    s = conical(start, end, n, m, coeff=1)
    return s


def conical(start, end, n, m=np.pi, coeff=1, bad_edge=False):
    r"""
    Generalized sampling function that extends from linear to more-than-cosine point distribution.
    The user selects the chord intervals over which this function is defined, the number of sampling points, and the "frequency" of the distribution.
    At a high level, this function translates a linear distribution of angles into a non-linear distribution of sampling points using a composite trigonometric function.
    The periodicity of the sampling refinement is defined by the angle sampling range ``m``.
    The cosine function turns a set of equally-spaced angles into a set of ``x`` coordinates denser at :math:`n\pi` (with n : int).
    The default :math:`m=\pi` thus means that sampled points will accumulate at the leading and trailing edge.
    If you double this frequency with :math:`m=2\pi`, then there will be an additional sampling concentration at mid chord, with :math:`m=3\pi`` there will be 2 additional concentrations, and so on.
    The user can also use a non-integer multiplier to have non-equally spaced refined areas - e.g. with :math:`m = \pi/2` the sampling will be coarse at the LE and refined at the TE.

    A more-than-cosine distribution exacerbates the non-linearity introduced by the cosine.
    The ``coeff`` parameter, ``b`` for conciseness in the code, defines the "strength" of the distortion:

        - :math:`b = 0`: linear distribution
        - :math:`b = 1`: cosine distribution
        - :math:`b > 1`: more-than-cosine distribution, meaning that the points are

    The overall function is composed of two sub-functions, continuous at :math:`b = 1 \rightarrow s = \cos(x)`.
    For :math:`b < 1`, the following function is used:

    .. math::

        s = \left(\frac{\cos(x) + 1}{2} - \frac{x}{\pi}\right)b + \frac{x}{\pi}

    While for coeff >=1:

    .. math::

        s = \frac{1}{2}\left(1 + \frac{\cos(x)}{\sqrt{\cos(x) ^ 2 + \sin(x) ^ 2 / b ^ 2}}\right)

    For more clarity, the user can plot these functions and see how the first one goes from linear to cosine as b approaches 1, and the second goes from cosine to a discontinuous (flipped) Heaviside function for :math:`b \rightarrow \infty`.
    Note that the cosine/conical functions are normalized and shifted to fit into the user-prescribed sampling interval.

    Parameters
    ----------
    start : float
        the airfoil chord location to start sampling at

    end : float
        the airfoil chord location to stop sampling at

    n : int
        the number of points to sample

    m : float
        the maximum angle used for sampling the point distribution, starting from zero.
        This implicitly defines the "frequency" of the refinement, e.g. m=pi refinement at LE and TE, :math:`m=2\pi` refinement at LE, TE, and mid-chord, etc

    bad_edge :  bool
        This is some kind of sneaky hack used to avoid bad meshes.
        If true, the second and second to last points of the ndarray are removed from the sampling vector.
        This is necessary to avoid bad mesh elements at small leading and trailing edges.
        Such problem often occurs for high N and high b.
        As a rule of thumb, the size of the smallest element (considering a normalized airfoil of size 1m) should always be > 1e-4 for pyHyp to extrude the mesh correctly.

    Returns
    -------
    s : Ndarray [N]
        The parametric spline locations that define the sampling

    """

    b = coeff

    if bad_edge is True:
        x = np.linspace(m, 0, n + 2)
        x = np.delete(np.delete(x, 1), -2)
    else:
        x = np.linspace(m, 0, n)
    if b >= 1:
        s = (1 + 1 / np.sqrt(np.cos(x) ** 2 + np.sin(x) ** 2 / b**2) * np.cos(x)) * 0.5
    else:
        cos = np.cos(x)
        s = ((cos + 1) / 2 - x[::-1] / np.pi) * b + x[::-1] / np.pi

    return s * (end - start) + start


def polynomial(start, end, n, m=np.pi, order=5):
    r"""
    similar to cosine spacing but instead of a unit circle, a function of the form :math:`1 - x^{\mathrm{order}}` is used.
    This does a better job on not overly clustering points at the edges.

    .. code-block:: text

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


    Parameters
    ----------
    start : float
        parametric location of the sampling start point

    end : float
        parametric location of the sampling end point

    n : int
        number of points to sample

    m : float
        the maximum angle for the sampling process

    order : float
        the order of polynomial from which to sample

    Returns
    -------
    s : Ndarray [N]
        The parametric spline locations that define the sampling
    """

    def poly(x, angle):
        return np.abs(x) ** order + np.tan(angle) * x - 1

    angles = np.linspace(m, 0, n)

    s = np.array([])
    for ang in angles:
        s = np.append(s, optimize.fsolve(lambda x: poly(x, ang), np.cos(ang))[0])

    return (s / 2 + 0.5) * (end - start) + start


def bigeometric(start, end, n, a1=0.001, b1=0.001, ra=1.1, rb=1.1):
    r"""
    This spacing function will create a distribution with a geometric sequence
    from both sides. It will try to find the optimal number of nodes to allocate
    to each sequence such that the middle region of constant spacings matches
    with the final spacing from each sequence. The default settings work well
    for :math:`n\sim\mathcal{O}(100)` (200 on entire airfoil).

    .. code-block:: text

         a1                           deltac                               b1
        |                             <----->                               |
        |  |   |    |     |     |     |     |     |     |     |    |   |  | |
        |                                                                   |
        <-- na=3 --><--------------- nc=n-2-na-nb -----------><--- nb=4  --->


    Parameters
    ----------
    start : float
        Parametric location of start of distribution (between 0 and 1).
    end : float
        Parametric location of end of distribution (between 0 and 1).
    n : int
        Number of nodes needed in distribution (including start and end).
    a1 : float
        Initial spacing from the start.
    b1 : float
        Initial spacing from the end.
    ra : float
        Geometric ratio from the start.
    rb : float
        Geometric ratio from the end.

    Returns
    -------
    s : Ndarray [N]
        The parametric spline locations that define the sampling
    """
    s = np.zeros(n)
    s[n - 1] = 1.0

    def findSpacing(na):
        a_na = a1 * ra**na
        nb = np.log(a_na / b1) / np.log(rb)
        nb = np.round(nb)
        da = a1 * (1 - ra**na) / (1 - ra)
        db = b1 * (1 - rb**nb) / (1 - rb)

        s_na = da
        s_nb = 1 - db
        dc = s_nb - s_na
        nc = n - (2 + na + nb)
        deltac = dc / (nc + 1)

        score = deltac / a_na - 1

        return score

    # Check to make sure spacing is not too large
    dc = 1.0 - a1 - b1
    nc = n - 4
    deltac = dc / (nc - 1)
    if deltac < a1 or deltac < b1:
        raise ValueError("Too many nodes. Decrease initial spacing.")

    # Find best spacing to get smooth distribution
    left = int(np.round(n * 0.01))
    right = int(n * 0.49)
    checkleft = findSpacing(left)
    checkright = findSpacing(right)

    if checkleft < 0 and checkright < 0:
        print(checkleft, checkright)
        raise ValueError("Try decreasing spacings")
    elif checkleft > 0 > checkright:
        na = optimize.bisect(findSpacing, left, right, xtol=1e-4, maxiter=100, disp=False)
    elif checkleft > 0 and checkright > 0:
        x0 = np.array([float(left)])
        opt = optimize.minimize(
            findSpacing, x0, method="tnc", bounds=[(left, right)], tol=1e-2, options={"maxiter": 1000}
        )
        na = opt.x

    # Compute final distribution
    na = int(np.round(na))
    a_na = a1 * ra**na
    nb = np.log(a_na / b1) / np.log(rb)
    nb = int(np.round(nb))
    da = a1 * (1 - ra**na) / (1 - ra)
    db = b1 * (1 - rb**nb) / (1 - rb)

    s_na = da
    s_nb = 1 - db
    dc = s_nb - s_na
    nc = n - (2 + na + nb)
    deltac = dc / (nc + 1)

    for i in range(1, n - 1):
        if i <= na:
            s[i] = s[i - 1] + a1 * ra ** (i - 1)
        elif i <= na + nc:
            s[i] = s[i - 1] + deltac
        else:
            j = n - i - 1
            s[i] = 1 - b1 * (1 - rb**j) / (1 - rb)

    s = s * (end - start) + start
    return s


def tanh_distribution(start, end, n, s0=None, s1=None):
    """
    Hyperbolic tangent distribution based on:
    https://www.cfd-online.com/Wiki/Structured_mesh_generation
    (Retrieved May 9, 2024)

    The original paper is:
    Marcel Vinokur. "On One-Dimensional Stretching Functions for Finite-Difference Calculations."
    Journal of Computational and Physics (1983)
    https://doi.org/10.1016/0021-9991(83)90065-7

    Parameters
    ----------
    start : float
        The location to start sampling at

    end : float
        The location to stop sampling at

    n : int
        The number of points to sample

    s0 : float
        The desired spacing at the start location

    s1 : float
        The desired spacing at the end location

    Returns
    -------
    dist : Ndarray [N]
        The parametric coordinates that define the distribution

    """

    if s0 is None or s1 is None:
        raise TypeError("s0 and s1 must be defined.")

    A = np.sqrt(s1 / s0)
    B = 1 / np.sqrt(s1 * s0)

    def func(delta):
        return B - np.sinh(delta) / delta

    delta = optimize.fsolve(func, 100)

    residual = B - np.sinh(delta) / delta

    if residual > 1e-4:
        raise ValueError("fsolve failed to converge.")

    xi = np.linspace(0, 1, n)
    u = 1 / 2 + np.tanh(delta * (xi - 1 / 2)) / (2 * np.tanh(delta / 2))
    dist = u / (A + (1 - A) * u) * (end - start) + start

    return dist


def joinedSpacing(n, spacingFunc=polynomial, func_args=None, s_LE=0.5):
    """
    Function that returns two point distributions joined at ``s_LE``.
    If it is desired to specify different spacing functions for the top and the bottom,
    the user can provide a list for the ``spacingFunc`` and ``func_args``.

    .. code-block::text
                            s1                            s2
        || |  |   |    |     |     |    |   |  | |||| |  |   |    |     |     |    |   |  | ||
                                                    /\
                                                    s_LE

    Note that one point is added when sampling due to the removal of "double"
    elements when returning the point array


    Parameters
    ----------
    n : int
        the number of points to sample

    spacingFunc : function
        a function that returns the sampling spacing

    func_args : dict
        options to pass into the spacingFunc

    s_LE : float
        parametric location of the leading edge

    Returns
    -------
    s : Ndarray [N]
        The parametric spline locations that define the sampling
    """
    if func_args is None:
        func_args = {}
    if callable(spacingFunc):
        spacingFunc = [spacingFunc] * 2
    if isinstance(func_args, dict):
        func_args = [func_args] * 2

    s1 = spacingFunc[0](0.0, s_LE, int(n * s_LE) + 1, **func_args[0])
    s2 = spacingFunc[1](s_LE, 1.0, int(n - n * s_LE) + 1, **func_args[1])

    # combine the two distributions
    s = np.append(s1[:], s2[1:])

    return s
