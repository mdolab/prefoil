import numpy as np
from scipy import optimize


def cosine(n, m=np.pi, **kwargs):
    s = conical(n, m, coeff=1)
    return s


def conical(n, m=np.pi, coeff=1, bad_edge=False):
    '''
    Here I make a sneaky hack to remove the second and second to last points
    of the ndarray. This is necessary to avoid bad mesh elements at small
    leading and trailing edges
    '''

    b=coeff

    if bad_edge is True:
        x = np.linspace(0, m, n + 2)
        x = np.delete(np.delete(x, 1), -2)
    else:
        x = np.linspace(0, m, n)
    if b >= 1:
        s = (1 + 1 / np.sqrt(np.cos(x)**2 + np.sin(x)**2 / b**2) * np.cos(x))*0.5
    else:
        cos = np.cos(x)
        s = ((cos+1)/2 - x[::-1]/np.pi)*b + x[::-1]/np.pi

    return s


def parabolic(n, m=np.pi, **kwargs):
    angles = np.linspace(0, m, n)
    s = np.array([])
    for ang in angles:
        if ang <= np.pi / 2:
            s = np.append(s, (-np.tan(ang) + np.sqrt(np.tan(ang)**2 + 4)) / 2)
        else:
            s = np.append(s, (-np.tan(ang) - np.sqrt(np.tan(ang)**2 + 4)) / 2)

    return s / 2 + 0.5


def polynomial(n, m=np.pi, coeff=5, **kwargs):

    order = coeff

    def func(x):
        return np.abs(x)**order + np.tan(ang) * x - 1
    angles = np.linspace(0, m, n)

    s = np.array([])
    for ang in angles:
        s = np.append(s, optimize.fsolve(func, np.cos(ang))[0])

    return s / 2 + 0.5


def joinedSpacing(n_up, spacingFunc_upr, coeff_upr, n_lwr, spacingFunc_lwr,
                  coeff_lwr, s_LE=0.5, bad_edge_upr=False, bad_edge_lwr=False,
                  **kwargs):
    """
    function that returns two point distributions joined at s_LE

                        s1                            s2
    || |  |   |    |     |     |    |   |  | |||| |  |   |    |     |     |    |   |  | ||
                                                /\
                                                s_LE

    Note that one point is added when sampling due to the removal of "double"
    elements when returning the point array

    """
    s1 = spacingFunc_upr(n=n_up+1, coeff=coeff_upr, bad_edge=bad_edge_upr,
                         m=np.pi, **kwargs) * s_LE
    s2 = spacingFunc_lwr(n=n_lwr+1,coeff=coeff_lwr, bad_edge=bad_edge_lwr,
                         m=np.pi, **kwargs) * (1 - s_LE) + s_LE

    return np.append(s2[1:],s1[1:])[::-1]