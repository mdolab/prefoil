"""
Sample usage script showing some use cases, and how the API should work.
"""


from pyfoil.pyFoil import Airfoil, readCoordFile
from pyfoil import sampling
import matplotlib.pyplot as plt

# import tecplot_interface as ti # This does not exist yet
from postprocessing.pytecplot.tecplotFileParser import TecplotParser

"""
Here we read an airfoil coordinate file from a database, perform geometric
cleanup, and then sample it with a particular distribution.
"""
filename = "../tests/rae2822.dat"
coords = readCoordFile(filename)
airfoil = Airfoil(coords, normalize=False)
coords = airfoil.getSampledPts(50, spacingFunc=sampling.conical, func_args={"coeff": 1})
fig1 = airfoil.plot()
fig1.suptitle("Test single distribution")
plt.show()

# # -------------------------------------------------------------------
"""
Here we sample the airfoil with two different distributions, and examine them side
by side.
"""

# airfoil = Airfoil(x=x,y=y)
# airfoil.derotate()

coords = airfoil.getSampledPts(50, spacingFunc=sampling.conical, func_args={"coeff": 1})
fig2 = airfoil.plot()
fig2.suptitle("coeff = 1")
coords2 = airfoil.getSampledPts(50, spacingFunc=sampling.conical, func_args={"coeff": 3})
fig3 = airfoil.plot()
fig3.suptitle("coeff = 3")
plt.show()

# -----------------------------------------
"""
Here we sample the airfoil with two separate distributions for the upper and
lower surfaces. We also thicken the trailing edge, and specify npts_TE during
sampling. These points are then saved to a plot3d file for use with pyHyp.
"""
# airfoil = Airfoil(X=X,cleanup=True)
# airfoil.thickenTE(thickness=0.01)

coords = airfoil.getSampledPts(
    50, spacingFunc=[sampling.bigeometric, sampling.polynomial], func_args=[{}, {"order": 8}]
)
airfoil.writeCoords(coords=coords, filename="new_coords", format="plot3d")

# Plotting samples
fig4 = airfoil.plot()
fig4.suptitle("Test double distribution")
plt.show()

exit()

# -----------------------------------------
"""
Here we read in a slice file from ADflow. There is some guesswork as to the
API for ti, but the idea is there. We retrieve geometric information from the airfoil,
cleanup the trailing edge, then compute geometric properties needed for postprocessing.
"""
filename = "tests/fc0_013_slices.dat"

# sectionName, sectionData, sectionConn = ti.readTecplotFEdata(filename)

# data = sectionData[1]
# conn = sectionConn[1]
# coords = ti.convert(data,conn)

# x = coords[:,0]
# y = coords[:,2]

tpp = TecplotParser(filename, debug=False)
# For plotting and indexing retrieve the variable names and zones
zoneNames = tpp.getZoneNames()
data = tpp.getData()
x = data[zoneNames[1]]["CoordinateX"]
y = data[zoneNames[1]]["CoordinateZ"]
plt.figure()
plt.plot(x, y, "-o")
plt.show()

## THIS APPROACH IS VERY BAD. NEED BETTER WAY TO DO GEOMETRY CLEANUP
airfoil = Airfoil(x=x, y=y, cleanup=True)
fig = airfoil.plotAirfoil()
airfoil.derotate()
airfoil.center()
airfoil.normalize()
fig2 = airfoil.plotAirfoil()
plt.show()
max_camber, camber_loc = airfoil.getMaxCamber()
max_thickness, thickness_loc = airfoil.getMaxThickness()
chord = airfoil.getChord()
c_c = camber_loc / chord
t_c = thickness_loc / chord

twist = airfoil.getTwist()
