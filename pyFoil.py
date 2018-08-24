"""
pyFoil
--------

Contains a class for creating, modifying and exporting airfoils.


Questions:
- Modes?!? Should we provide any functionality for that?
"""
from __future__ import print_function
from __future__ import division

import warnings
import numpy as np
import pyspline
from scipy.optimize import fsolve, brentq


def _readSliceFile(filename):
    pass

def _readCoordFile(filename):
    pass

def _reorder(coords):
    pass

def _genNACACoords(name):
    pass


def _writePlot3D(filename,x,y):
    if '.' not in filename:
        filename = filename + '.fmt'
    f = open(filename, 'w')
    f.write('1\n')
    f.write('%d %d %d\n'%(len(x), 2, 1))
    for iDim in range(3):
        for j in range(2):
            for i in range(len(x)):
                if iDim == 0:
                    f.write('%g\n'%x[i])
                elif iDim == 1:
                    f.write('%g\n'%y[i])
                else:
                    f.write('%g\n'%(float(j)))
    f.close()


class Airfoil(object):
    """
    Create an instance of an airfoil. There are two ways of instantiating 
    this object: by passing in a set of points, or by reading in a Tecplot
    .dat file. 

    Parameters
    ----------
    k : int
        Order for spline
    """
    def __init__(self, **kwargs):
        pass


    def recompute(self):
        pass
    


## Geometry Information
    def getLE(self):
        pass

    def getLERadius(self):
        pass

    def getTEAngle(self):
        pass
    
    def getTwist(self):
        pass
    
    def getCamber(self):
        pass
    
    def getMaxCamber(self):
        pass
    
    def getThickness(self):
        pass

    def getMaxThickness(self):
        pass
    
    def getChord(self):
        pass
    
    def getChordLine(self):
        pass

    def isReflex(self):
        pass
    def isSymmetric(self):
        pass




## Geometry Modification


    def _derotate(self):
        pass

    def _normalize(self):
        pass
    
    def _center(self):
        pass

    def smooth(self,method):
        pass
    
    def thickenTE(self):
        pass
    
    def sharpenTE(self):
        pass

    def roundTE(self):
        pass


    def _removeTEPts(self):
        pass

    def _translateCoords(self):
        pass
    def _rotateCoords(self):
        pass
    def _scaleCoords(self):
        pass

## Sampling

'''
Check cosine spacing, OAS has a parameter which varies from 0 to 1,
corresponding to linear and cosine sampling. Maybe add that.

If all three are elliptic with specific parameters, we should call that from 
the other routines rather than having duplicate code.
'''
    def _cosSpacing(n, m=np.pi):
        x = np.linspace(0, m, n)
        s = np.cos(x)
        return s / 2 + 0.5


    def _ellipticalSpacing(n,  b=1,  m=np.pi):
        x = np.linspace(0, m, n)
        s = 1 + b / np.sqrt(np.cos(x)**2 * b**2 + np.sin(x)**2) * np.cos(x)
        return s * b


    def _parabolicSpacing(n, m=np.pi):
        # angles = np.linspace(0, m, (n + 1) // 2)
        angles = np.linspace(0, m, n)
        # x = np.linspace(1, -1, n)
        s = np.array([])
        for ang in angles:
            if ang <= np.pi / 2:
                s = np.append(s, (-np.tan(ang) + np.sqrt(np.tan(ang)**2 + 4)) / 2)
            else:
                s = np.append(s, (-np.tan(ang) - np.sqrt(np.tan(ang)**2 + 4)) / 2)

        # print 's', s, -1 * s[-2::-1]
        # s = np.append(s, -1 * s[-2::-1])[::-1]
        return s / 2 + 0.5


    def _polynomialSpacing(n, m=np.pi, order=5):

        def func(x):
            return np.abs(x)**order + np.tan(ang) * x - 1
        angles = np.linspace(0, m, n)

        s = np.array([])
        for ang in angles:
            s = np.append(s, fsolve(func, np.cos(ang))[0])

        return s / 2 + 0.5


    def _joinedSpacing(n, spacingFunc, s_LE=0.5, equalPts=False, repeat=False, **kwargs):
        """
        function that returns two point distributions joined at s_LE

                            s1                            s2
        || |  |   |    |     |     |    |   |  | |||| |  |   |    |     |     |    |   |  | ||
                                                    /\
                                                    s_LE

        """
        offset1 = np.pi / (n * s_LE)
        if repeat:
            offset2 = 0
        else:
            offset2 = np.pi / (n * s_LE)

        if equalPts:
            ns1 = n * .5
            ns2 = ns1
        else:
            ns1 = n * s_LE
            ns2 = n * (1 - s_LE)

        s1 = spacingFunc(ns1, m=np.pi - offset1, **kwargs) * s_LE
        s2 = spacingFunc(ns2, m=np.pi - offset2, **kwargs) * (1 - s_LE) + s_LE

        return np.append(s2, s1)[::-1]

    def sample(self,distribution,npts,**kwargs):
        pass

## Output
    def genCoords(self, filename):
        pass
    def writeTecplot(self):
        pass


## Utils
    def plotAirfoil(self):
        pass