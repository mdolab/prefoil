from __future__ import print_function
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

def bigeometric(start, end, n, a1=0.001, b1=0.001, ra=1.1, rb=1.1):
    """
    This spacing function will create a distribution with a geometric sequence
    from both sides. It will try to find the optimal number of nodes to allocate
    to each sequence such that the middle region of constant spacings matches
    with the final spacing from each sequence. The default settings work well
    for n~100 (200 on entire airfoil).

     a1                           deltac                               b1
    |                             <----->                               |
    |  |   |    |     |     |     |     |     |     |     |    |   |  | |
    |                                                                   |
    <-- na=3 --><--------------- nc=n-2-na-nb -----------><--- nb=4  --->
    Parameters
    ----------
    start : int
        Parametric location of start of distribution (between 0 and 1).
    end : int
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
    """
    s = np.zeros(n)
    s[n-1] = 1.0

    def findSpacing(na, search=False):
        a_na = a1 * ra**na
        nb = np.log(a_na / b1) / np.log(rb)
        nb = round(nb)
        b_nb = b1 * rb**nb
        da = a1*(1 - ra**na) / (1 - ra)
        db = b1*(1 - rb**nb) / (1 - rb)

        s_na = da
        s_nb = 1 - db
        dc = s_nb - s_na
        nc = n - (2 + na + nb)
        deltac = dc / (nc+1)

        score = deltac/a_na - 1
        if search:
            # print(na, nc, nb, a_na, b_nb, deltac)
            return score
        else:
            return score

    # Check to make sure spacing is not too large
    dc = 1.0 - a1 - b1
    nc = n - 4
    deltac = dc / (nc-1)
    if deltac < a1 or deltac < b1:
        print('Too many nodes. Decrease initial spacing.')
        exit()

    # Find best spacing to get smooth distribution
    # print('Finding optimal bigeometric spacing...')
    left = int(round(n*0.01))
    right = int(n*0.49)
    checkleft = findSpacing(left)
    checkright = findSpacing(right)

    if checkleft < 0 and checkright < 0:
        print(checkleft, checkright, 'Try decreasing spacings')
        exit()
    elif checkleft > 0 and checkright < 0:
        # print('Bisection method')
        na = optimize.bisect(findSpacing, left, right, (True), xtol=1e-4,
            maxiter=100, disp=False)
    elif checkleft > 0 and checkright > 0:
        # print('Minimize method')
        x0 = np.array([float(left)])
        opt = optimize.minimize(findSpacing, x0, (True), method='tnc',
            bounds=[(left, right)], tol=1e-2, options={'maxiter':1000})
        na = opt.x

    # import matplotlib.pyplot as plt
    # x = np.linspace(left, right, 100)
    # y = np.zeros_like(x)
    # for i in range(len(x)):
    #     y[i] = findSpacing(x[i])
    # plt.figure()
    # plt.plot(x, y)
    # plt.plot(na, findSpacing(na), 'rx')
    # plt.show()

    # Compute final distribution
    na = int(round(na))
    a_na = a1 * ra**na
    nb = np.log(a_na / b1) / np.log(rb)
    nb = int(round(nb))
    b_nb = b1 * rb**nb
    da = a1*(1 - ra**na) / (1 - ra)
    db = b1*(1 - rb**nb) / (1 - rb)

    s_na = da
    s_nb = 1 - db
    dc = s_nb - s_na
    nc = n - (2 + na + nb)
    deltac = dc / (nc+1)
    # print('Score:', deltac/a_na, deltac/b_nb)

    for i in range(1, n-1):
        if i <= na:
            s[i] = s[i-1] + a1*ra**(i-1)
        elif i <= na + nc:
            s[i] = s[i-1] + deltac
        else:
            j = n - i - 1
            s[i] = 1 - b1*(1 - rb**j) / (1 - rb)

    s = s * (end - start) + start
    return s

def joinedSpacing(n, spacingFunc=polynomial, func_args={}, s_LE=0.5):
    """
    Function that returns two point distributions joined at s_LE. If it is
    desired to specify different spacing functions for the top and the bottom,
    the user can provide a list for the spacingFunc and func_args.

                        s1                            s2
    || |  |   |    |     |     |    |   |  | |||| |  |   |    |     |     |    |   |  | ||
                                                /\
                                                s_LE

    Note that one point is added when sampling due to the removal of "double"
    elements when returning the point array
    """
    if callable(spacingFunc):
        spacingFunc = [spacingFunc]*2
    if isinstance(func_args, dict):
        func_args = [func_args]*2

    s1 = spacingFunc[0](0., s_LE, int(n * s_LE) + 1, **func_args[0])
    s2 = spacingFunc[1](s_LE, 1., int(n - n * s_LE) + 1, **func_args[1])

    # combine the two distributions
    s = np.append(s1[:], s2[1:])

    return s
