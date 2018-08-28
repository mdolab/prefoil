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
from numpy.linalg import norm 


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



def _readCoordFile(filename):
    """ Load the airfoil file"""
    f = open(filename, 'r')
    line  = f.readline() # Read (and ignore) the first line
    r = []
    try:
        r.append([float(s) for s in line.split()])
    except:
        r = []

    while True:
        line = f.readline()
        if not line:
            break # end of file
        if line.isspace():
            break # blank line
        r.append([float(s) for s in line.split()])

    X = np.array(r)
    return X

def _reorder(coords):
    pass

def _genNACACoords(name):
    pass

def _cleanup_TE(X,tol):
    TE = np.mean(X[[-1,0],:],axis=0)
    return X, TE

def _writePlot3D(filename,X):
    if '.' not in filename:
        filename = filename + '.fmt'
    x = X[:,0]
    y = X[:,1]
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
        The file name containing the airfoil coordinates
    """
    def __init__(self, **kwargs):
        self.TE = None
        self.LE = None
        self.s_LE = None
        self.LE_rad = None
        self.TE_angle = None
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

        self.X, self.TE = _cleanup_TE(self.X,tol=1e-3)
        
        if 'cleanup' in kwargs and kwargs['cleanup']:
            self._cleanup()
        self.recompute()


    def recompute(self):
        self.spline = Curve(X=self.X,k=self.k,nCtl=self.nCtl)

## Geometry Information
    def getLE(self):
        '''
        Calculates the leading edge point on the spline, which is defined as the point furthest away from the TE. The spline is assumed to start at the TE. The routine uses a root-finding algorithm to compute the LE.

        Let the TE be at point :math:`x_0, y_0`, then the Euclidean distance between the TE and any point on the airfoil spline is :math:`\ell(s) = \sqrt{\Delta x^2 + \Delta y^2}`, where :math:`\Delta x = x(s)-x_0` and :math:`\Delta y = y(s)-y_0`. We know near the LE, this quantity is concave. Therefore, to find its maximum, we differentiate and use a root-finding algorithm on its derivative. 
        :math:`\\frac{\mathrm{d}\ell}{\mathrm{d}s} = \\frac{\Delta x\\frac{\mathrm{d}x}{\mathrm{d}s} + \Delta y\\frac{\mathrm{d}y}{\mathrm{d}s}}{\ell}`

        The function dellds computes the quantity :math:`\Delta x\\frac{\mathrm{d}x}{\mathrm{d}s} + \Delta y\\frac{\mathrm{d}y}{\mathrm{d}s}` which is then used by brentq to find its root, with an initial bracket at [0.3, 0.7].

        TODO
        Use a Newton solver, employing 2nd derivative information and use 0.5 as the initial guess. 
        '''
        if self.s_LE is None:
            def dellds(s,spline,TE):
                pt = spline.getValue(s)
                deriv = spline.getDerivative(s)
                dx = pt[0] - TE[0]
                dy = pt[1] - TE[1]
                return dx * deriv[0] + dy * deriv[1]

            self.s_LE = brentq(dellds,0.3,0.7,args=(self.spline,self.TE))
            self.LE = self.spline.getValue(self.s_LE)
        return self.s_LE

    def getLERadius(self):
        '''
        Computes the leading edge radius of the airfoil. Note that this is heavily dependent on the initialization points, as well as the spline order/smoothing.
        '''
        if self.s_LE is None:
            self.getLE()
        first = self.spline.getDerivative(self.s_LE)
        second = self.spline.getSecondDerivative(self.s_LE)
        self.LE_rad = norm(first)**3 / norm(first[0]*second[1] - first[1]*second[0])
        return self.LE_rad

    def getTEAngle(self):
        '''
        Computes the trailing edge angle of the airfoil. We assume here that the spline goes from top to bottom, and that s=0 and s=1 corresponds to the 
        top and bottom trailing edge points. Whether or not the airfoil is closed is irrelevant. 
        '''
        top = self.spline.getDerivative(0)
        top = top/norm(top)
        bottom = self.spline.getDerivative(1)
        bottom = bottom/norm(bottom)
        print(np.dot(top,bottom))
        self.TE_angle = np.pi - np.arccos(np.dot(top,bottom))
        return np.rad2deg(self.TE_angle)
    
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
        if self.twist is not None:
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
        if self.chord is not None:
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
        if self.LE is not None:
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

    def _getDefaultSampling(self,npts = 100):
        sampling = np.linspace(0,1,npts)
        return self.spline.getValue(sampling)
## Output
    def writeCoords(self, filename,fmt='plot3d'):
        pass

## Utils
    def plotAirfoil(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        pts = self._getDefaultSampling(npts=1000)
        plt.plot(pts[:,0],pts[:,1],'-')
        plt.axis('equal')
        # pt = self.LE + np.array([self.LE_rad,0])
        # circle = plt.Circle(pt, self.LE_rad, color='r',fill=False)
        # ax.add_artist(circle)
        # plt.plot(self.LE[0],self.LE[1],'ok')
        plt.show()