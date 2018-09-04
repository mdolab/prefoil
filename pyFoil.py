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
import sampling as smp
from numpy.linalg import norm
import os
import pygeo
import matplotlib.pyplot as plt

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

def _cleanup_pts(X):
    '''
    For now this just removes points which are too close together. In the future we may need to add further
    functionalities. This is just a generic cleanup tool which is called as part of preprocessing.
    DO NOT USE THIS, IT CURRENTLY DOES NOT WORK
    '''
    uniquePts, link = pygeo.geo_utils.pointReduce(X, nodeTol=1e-12)
    nUnique = len(uniquePts)

    # Create the mask for the unique data:
    mask = np.zeros(nUnique, 'intc')
    for i in range(len(link)):
        mask[link[i]] = i

    # De-duplicate the data
    data = X[mask, :]
    return data


def _reorder(coords):
    '''
    This function serves two purposes. First, it makes sure the points are oriented in counter-clockwise
    direction. Second, it makes sure the points start at the TE. 
    '''
    pass

def _genNACACoords(name):
    pass

def _cleanup_TE(X,tol):
    TE = np.mean(X[[-1,0],:],axis=0)
    return X, TE

def _writePlot3D(filename,x,y):
    filename += '.fmt'
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

def _writeDat(filename,x,y):
    filename += '.dat'
    f = open(filename, 'w')

    for i in range(0, len(x)):
        f.write(str(round(x[i], 12)) + "\t\t"
                + str(round(y[i], 12)) + '\n'
                )
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

def checkCellRatio(X,ratio_tol=1.2):
    X_diff = X[1:,:] - X[:-1,:]
    cell_size = np.sqrt(X_diff[:,0]**2 + X_diff[:,1]**2)
    crit_cell_size = np.flatnonzero(cell_size<1e-10)
    for i in crit_cell_size:
        print("critical I", i)
    
    cell_ratio = cell_size[1:]/cell_size[:-1]
    exc = np.flatnonzero(cell_ratio > ratio_tol)

    if exc.size > 0:
        print('WARNING: There are ', exc.size, ' elements which exceed '
                                               'suggested cell ratio: ',
              exc)

    max_cell_ratio = np.max(cell_ratio, 0)
    avg_cell_ratio = np.average(cell_ratio, 0)
    print('Max cell ratio: ', max_cell_ratio)
    print('Average cell ratio', avg_cell_ratio)

    return cell_ratio, max_cell_ratio, avg_cell_ratio, exc


class Airfoil(object):
    """
    Create an instance of an airfoil. There are two ways of instantiating 
    this object: by passing in a set of points, or by reading in a coordinate
    file. The points need not start at the TE nor go ccw around the airfoil,
    but they must be ordered such that they form a continuous airfoil surface.
    If they are not (due to MPI or otherwise), use the order() function within
    tecplotFileParser.

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
    name : str
        The name of the airfoil. Default will be the filename, if one is supplied.
    """
    def __init__(self, **kwargs):
        self.TE = None
        self.LE = None
        self.s_LE = None
        self.LE_rad = None
        self.TE_angle = None
        self.twist = None
        self.chord = None
        self.chord_vec = None
        self.camber = None
        self.top = None
        self.bottom = None
        self.sampled_X = None
        self.camber_pts = None

        if 'k' in kwargs:
            self.k = kwargs['k']
        else:
            self.k = 3
        if 'nCtl' in kwargs:
            self.nCtl = kwargs['nCtl']
        else:
            self.nCtl = 20
        if 'name' in kwargs:
            self.name = kwargs['name']
        elif 'filename' in kwargs:
            self.name = kwargs['filename'].split('.')[0]
        else:
            self.name = 'unnamed airfoil'

        if 'X' in kwargs:
            self.X = kwargs['X']
        elif 'x' in kwargs and 'y' in kwargs:
            self.X = np.stack((kwargs['x'],kwargs['y']),axis=1)
        elif 'filename' in kwargs:
            self.X = _readCoordFile(kwargs['filename'])
        else:
            raise Error('You need to provide either points or a filename to initialize.')

        #self.X = _cleanup_pts(self.X)

        self.X, self.TE = _cleanup_TE(self.X,tol=1e-3)
        if 'cleanup' in kwargs and kwargs['cleanup']:
            self._cleanup()
        self.recompute()

    def recompute(self):
        self.spline = Curve(X=self.X,k=self.k) #nCtl=self.nCtl

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
        return self.LE


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

    def getTE(self):
        self.TE = (self.spline.getValue(0) + self.spline.getValue(1))/2
        return self.TE

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
    
    def _splitAirfoil(self):
        if self.s_LE is None:
            self.getLE()
        self.top, self.bottom = self.spline.splitCurve(self.s_LE)

    def getTwist(self):
        if self.chord_vec is None:
            self.getChord()
        
        normalized_chord = self.chord_vec / self.chord
        self.twist = np.arccos(normalized_chord.dot(np.array([1., 0.]))) * np.sign(normalized_chord[1])
        return self.twist
    
    def getCTDistribution(self):
        '''
        Return the coordinates of the camber points, as well as the thicknesses (this is with british
        convention).
        '''
        self._splitAirfoil()
        if self.twist is None:
            self.getTwist()
        if self.chord is None:
            self.getChord
        num_chord_pts = 100
        
        # Compute the chord
        chord_pts = np.vstack([self.LE,self.TE])
        chord = line(chord_pts)

        cos_sampling = np.linspace(0,1,num_chord_pts,endpoint=False)  # [1:] +1
        chord_pts = chord.getValue(cos_sampling)
        camber_pts = np.zeros((num_chord_pts,2))
        thickness_pts = np.zeros((num_chord_pts,2))
        for j in range(chord_pts.shape[0]):
            direction = np.array([np.cos(np.pi/2 - self.twist), np.sin(np.pi/2 - self.twist)])
            direction = direction/norm(direction)
            top = chord_pts[j,:] + 0.5*self.chord * direction
            bottom = chord_pts[j,:] - 0.5*self.chord * direction
            temp = np.vstack((top,bottom))
            normal = line(temp)
            s_top,t_top,D = self.top.projectCurve(normal,nIter=5000,eps=1e-16)
            s_bottom,t_bottom,D = self.bottom.projectCurve(normal,nIter=5000,eps=1e-16)
            intersect_top = self.top.getValue(s_top)
            intersect_bottom = self.bottom.getValue(s_bottom)

            # plt.plot(temp[:,0],temp[:,1],'-og')
            # plt.plot(intersect_top[0],intersect_top[1],'or')
            # plt.plot(intersect_bottom[0],intersect_bottom[1],'ob')

            camber_pts[j,:] = (intersect_top + intersect_bottom)/2
            thickness_pts[j,0] = (intersect_top[0] + intersect_bottom[0])/2
            thickness_pts[j,1]= (intersect_top[1] - intersect_bottom[1]) / 2
        # plt.plot(camber_pts[:,0],camber_pts[:,1],'ok')

        self.camber_pts = np.vstack((self.LE,camber_pts,self.TE)) # Add TE and LE to the camber points.
        self.thickness_pts = np.vstack((self.LE,thickness_pts,self.TE))

        return self.camber_pts, self.thickness_pts

    def getMaxCamber(self):
        pass
    

    def getMaxThickness(self,method):
        '''
        method : str
            Can be one of 'british', 'american', or 'projected'
        '''
        pass
    
    def getChord(self):
        if self.LE is None:
            self.getLE()
        if self.TE is None:
            self.getTE()
        self.chord_vec = self.TE - self.LE
        self.chord = norm(self.chord_vec)
        return self.chord

    def getChordVec(self):
        if self.chord_vec is None:
            self.getChord()
        return self.chord_vec

    def isReflex(self):
        '''
        An airfoil is reflex if the derivative of the camber line at the trailing edge is positive.
        #TODO this has not been tested
        '''
        if self.camber is None:
            self.getCamber()
        
        if self.camber.getDerivative(1)[1] > 0:
            return True
        else:
            return False

    def isSymmetric(self, tol=1e-6):
        pass




## Geometry Modification
    def _cleanup(self,derotate=True, normalize=True, center=True):
        if derotate or normalize or center:
            self.recompute()
            origin = np.zeros(2)
            sample_pts = self._getDefaultSampling()
            self.getLE()
            self.getTwist()
            self.getChord()
            '''
            Order of operation here is important, even though all three operations are linear, because
            we rotate about the origin for simplicity.
            '''
            if center:
                delta = -1.0*self.LE
                sample_pts = _translateCoords(sample_pts,delta)
            if derotate:
                angle = -1.0*self.twist
                sample_pts = _rotateCoords(sample_pts,angle,origin)
            if normalize:
                factor = 1.0/self.chord
                sample_pts = _scaleCoords(sample_pts,factor,origin)
            
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
        elif np.all(self.LE == np.zeros(2)):
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

    def _removeTEPts(self):
        pass

## Sampling
    def sample(self, upper, lower=None, npts_TE=None, cell_check=True):
        '''
        This function defines the point sampling along airfoil surface.
        An example dictionary is reported below:

        >>> sample_dict = {'distribution' : 'conical',
        >>>        'coeff' : 1,
        >>>        'npts' : 50,
        >>>        'bad_edge': False}

        The point distribution currently implemented are:
            - *Cosine*:
            - *Conical*:
            - *Parabolic*:
            - *Polynomial*:

        :param upper: dictionary
                Upper surface sampling dictionary
        :param lower: dictionary
                Lower surface sampling dictionary
        :param npts_TE: float
                Number of points along the **blunt** trailing edge
        :return: Coordinates array, anticlockwise, from trailing edge
        '''
        single_distr = False
        points_init = upper['npts']
        if lower is None:
            single_distr = True
            upper['npts'] = upper['npts']//2
            lower = upper
        else:
            points_init_lwr = len(lower['npts'])
        if self.s_LE is None:
            self.getLE()

        bad_edge_upr = False
        bad_edge_lwr = False
        if 'bad_edge' in upper:
            bad_edge_upr = upper['bad_edge']
        if 'bad_edge' in lower:
            bad_edge_lwr = lower['bad_edge']

        upper_distr = getattr(smp, upper['distribution'])
        lower_distr = getattr(smp, lower['distribution'])

        sampling = smp.joinedSpacing(upper['npts'],upper_distr,
                                 upper['coeff'],lower['npts'],
                                 lower_distr,lower['coeff'],
                                 s_LE=self.s_LE,bad_edge_upr=bad_edge_upr,
                                 bad_edge_lwr=bad_edge_lwr)

        coords = self.spline.getValue(sampling)

        # Adding last point (1,-0) for pyHyp issues
        # TODO: Add handling of TE, esp blunt or round
        end_point = np.copy(coords[0])
        if end_point[1] == 0.0:
            end_point[1] = -0.0
        coords = np.concatenate((coords, end_point.reshape(1, -1)), axis=0)
        if cell_check is True:
            checkCellRatio(coords)
        self.sampled_X = coords
        x = coords[:,0]
        y = coords[:,1]

        # To be updated later on if new point add/remove operations are included
        # len(x)-1 because of the last point added for "closure"
        if single_distr is True and len(x)-1 != points_init:
            print('WARNING: The number of sampling points has been changed \n'
                    '\t\tCurrent points number: %i' % (len(x)))
        return x, y

    def _getDefaultSampling(self,npts = 100):
        sampling = np.linspace(0,1,npts)
        return self.spline.getValue(sampling)
## Output
    def writeCoords(self, filename,fmt='plot3d'):

        if self.sampled_X is not None:
            x = self.sampled_X[:,0]
            y = self.sampled_X[:,1]
        else:
            '''
            We have to discuss which types of printfiles we want to get and how
            to handle them (does this class print the last "sampled" x,y or do 
            we want more options?)
            '''
            raise Error("No coordinates to print, run .sample() first")

        if fmt == 'plot3d':
            _writePlot3D(filename, x, y)
        elif fmt == 'dat':
            _writeDat(filename, x, y)
        else:
            print(fmt)
            raise Warning('Output file not supported')

## Utils
# maybe remove and put into a separate location?
    def plotAirfoil(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        pts = self._getDefaultSampling(npts=1000)
        plt.plot(pts[:,0],pts[:,1],'-')
        plt.axis('equal')
        if self.sampled_X is not None:
            plt.plot(self.sampled_X[:,0],self.sampled_X[:,1],'-o')
        if self.camber_pts is not None:
            plt.plot(self.camber_pts[:,0],self.camber_pts[:,1],'-o')

        plt.title(self.name)
        return fig