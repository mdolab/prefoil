from pyFoil import Airfoil

filename = 'naca0012.dat'
airfoil = Airfoil(filename=filename,derotate=True,normalize=True)

airfoil.smooth(method='Laplacian')
x,y = airfoil.sample(distributions=['conical','polynomial'],params=[{'b':1},{'order':5}],npts=[50,70])

# -------------------------------------------------------------------


airfoil = Airfoil(x=x,y=y,derotate=True,norm=True)
airfoil.normalize()

airfoil.smooth(method='Laplacian')
x,y = airfoil.sample(distribution='conical',b=1, npts=100)
x2,y2 = airfoil.sample(distribution='conical',b=3, npts=50)


#-----------------------------------------

airfoil = Airfoil(x=x,y=y,derotate=True,norm=True)
airfoil.thickenTE(TE_thickness=0.01)
x,y = airfoil.sample(distributions=['conical','polynomial'],params=[{'b':1},{'order':5}],npts=[50,70],npts_TE=17)

airfoil.writeCoords('new_coords.txt',fmt='plot3d')