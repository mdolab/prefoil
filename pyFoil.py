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
import sampling


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
        '''
        Calculates the leading edge point on the spline, which is defined as the point furthest away from the TE. The spline is assumed to start at the TE. The routine uses a root-finding algorithm to compute the LE.

        Let the TE be at point :math:`x_0, y_0`, then the Euclidean distance between the TE and any point on the airfoil spline is :math:`\ell(s) = \sqrt{\Delta x^2 + \Delta y^2}`, where :math:`\Delta x = x(s)-x_0` and :math:`\Delta y = y(s)-y_0`. We know near the LE, this quantity is concave. Therefore, to find its maximum, we differentiate and use a root-finding algorithm on its derivative.
        :math:`\\frac{\mathrm{d}\ell}{\mathrm{d}s} = \\frac{\Delta x\\frac{\mathrm{d}x}{\mathrm{d}s} + \Delta y\\frac{\mathrm{d}y}{\mathrm{d}s}}{\ell}`

        The function dellds computes the quantity :math:`\Delta x\\frac{\mathrm{d}x}{\mathrm{d}s} + \Delta y\\frac{\mathrm{d}y}{\mathrm{d}s}` which is then used by brentq to find its root, with an initial bracket at [0.3, 0.7].
        '''

        spline = self.spline

        def dellds(s, spline, TE):
            pt = spline.getValue(s)
            deriv = spline.getDerivative(s)
            dx = pt[0] - TE[0]
            dy = pt[1] - TE[1]
            return dx * deriv[0] + dy * deriv[1]

        TE = spline.getValue(0)
        s_LE = optimize.brentq(dellds, 0.3, 0.7, args=(spline, TE))

        self.s_LE = s_LE

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

    def sample(self,distribution,npts,**kwargs):

        self.point_distribution =
        pass

## Output
    def genCoords(self, filename):
        pass
    def writeTecplot(self):
        pass


## Utils
    def plotAirfoil(self):
        pass