"""
pyFoil
--------

Contains a class for creating, modifying and exporting airfoils.


Questions:
- Modes?!? Should we provide any functionality for that?
- Do we want twist in deg or rad?
"""
from __future__ import print_function
from __future__ import division

import warnings
import numpy as np
from pyspline.python.pySpline import Surface, Curve, line
from scipy.optimize import fsolve, brentq

class Error(Exception):
    """
    Format the error message in a box to make it clear this
    was a explicitly raised exception.
    """
    def __init__(self, message):
        msg = '\n+' + '-'*78 + '+' + '\n' + '| pySpline Error: '
        i = 17
        for word in message.split():
            if len(word) + i + 1 > 78: # Finish line and start new one
                msg += ' '*(78-i)+'|\n| ' + word + ' '
                i = 1 + len(word)+1
            else:
                msg += word + ' '
                i += len(word)+1
        msg += ' '*(78-i) + '|\n' + '+'+'-'*78+'+'+'\n'
        print(msg)
        Exception.__init__()


def _getDefaultSampling():
    pass

def _readCoordFile(filename):
    pass

def _reorder(coords):
    pass

def _genNACACoords(name):
    pass

def _cleanup_TE(X,tol):
    pass

def _writePlot3D(filename,X):
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

def _translateCoords(X,dX):
    """shifts the input coordinates by dx and dy"""
    return X + dX


def _rotateCoords(X, angle, origin):
    """retruns the coordinates rotated about the specified origin by angle (in deg)"""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c,-s), (s, c)))
    shifted_X = X - origin
    shifted_rotated_X = np.dot(shifted_X,R.T)
    rotated_X = shifted_rotated_X + origin
    return rotated_X


def _scaleCoords(X, scale, origin):
    """scales the coordinates in both dimension by the scaling factor"""
    shifted_X = X - origin
    shifted_scaled_X = shifted_X * scale
    scaled_X = shifted_scaled_X + origin
    return scaled_X


class Airfoil(object):
    """
    Create an instance of an airfoil. There are two ways of instantiating 
    this object: by passing in a set of points, or by reading in a coordinate
    file. 

    Parameters
    ----------
    k : int
        Order of the spline
    nCtl : int
        Number of control points
    X : ndarray[N,2]
        Full array of airfoil coordinates
    x : ndarray[N]
        Just x coordinates of the airfoil
    y : ndarray[N]
        Just y coordinates of the airfoil
    filename : str
        The filename containing the airfoil coordinates
    """
    def __init__(self, **kwargs):
        self.TE = None
        self.LE = None
        self.s_TE = None
        self.s_LE = None
        self.LE_rad = None
        self.TE_ang = None
        self.twist = None
        self.chord = None

        if 'k' in kwargs:
            self.k = kwargs['k']
        else:
            self.k = 3
        if 'nCtl' in kwargs:
            self.nCtl = kwargs['nCtl']
        else:
            self.nCtl = 20

        if 'X' in kwargs:
            self.X = kwargs['X']
        elif 'x' in kwargs and 'y' in kwargs:
            self.X = np.hstack((kwargs['x'],kwargs['y']))
        elif 'filename' in kwargs:
            self.X = _readCoordFile(kwargs['filename'])
        else:
            raise Error('You need to provide either points or a filename to initialize.')

        self.X = _cleanup_TE(self.X)
        
        if 'cleanup' in kwargs and kwargs['cleanup']:
            self._cleanup()
        self.recompute()

    def recompute(self):
        self.spline = Curve(X=self.X,k=self.k,nCtl=self.nCtl)
    


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
    def _cleanup(self,derotate=True, normalize=True, center=True):
        if derotate or normalize or center:
            self.recompute()
            origin = np.zeros(2)
            sample_pts = self._getDefaultSampling()
            if derotate:
                angle = -1.0*self.getTwist()
                sample_pts = _rotateCoords(sample_pts,angle,origin)
            if normalize:
                factor = 1.0/self.getChord()
                sample_pts = _scaleCoords(sample_pts,factor,origin)
            if center:
                delta = -1.0*self.getLE()
                sample_pts = _translateCoords(sample_pts,delta)
            
            self.X = sample_pts
            self.recompute()

    def rotate(self,angle,origin=np.zeros(2)):
        sample_pts = self._getDefaultSampling()
        self.X = _rotateCoords(sample_pts,angle,origin)
        self.recompute()
        self.twist += angle

    def derotate(self,origin=np.zeros(2)):
        if self.spline is None:
            self.recompute()
        if self.twist is None:
            self.getTwist()
        elif self.twist == 0:
            return
        self.rotate(-1.0*self.twist,origin=origin)
    
        
    def scale(self,factor,origin=np.zeros(2)):
        sample_pts = self._getDefaultSampling()
        self.X = _scaleCoords(sample_pts,factor,origin)
        self.recompute()
        self.chord *= factor

    def normalize(self,origin=np.zeros(2)):
        if self.spline is None:
            self.recompute()
        if self.chord is None:
            self.getChord()
        elif self.chord == 1:
            return
        self.scale(1.0/self.chord,origin=origin)
    
    def translate(self,delta):
        sample_pts = self._getDefaultSampling()
        self.X = _translateCoords(sample_pts,delta)
        self.recompute()
        self.LE += delta

    def center(self):
        if self.spline is None:
            self.recompute()
        if self.LE is None:
            self.getChord()
        elif self.LE == np.zeros(2):
            return
        self.translate(-1.0*self.LE)

    def smooth(self,method):
        pass
    
    def thickenTE(self):
        pass
    
    def sharpenTE(self):
        pass

    def roundTE(self):
        pass

## Sampling
    def sample(self,*args,**kwargs):
        pass

    def _getDefaultSampling(self):
        pass
## Output
    def writeCoords(self, filename,fmt='plot3d'):
        pass

## Utils
    def plotAirfoil(self):
        pass