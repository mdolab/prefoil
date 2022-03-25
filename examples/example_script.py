"""
Sample usage script showing some use cases, and how the API should work.
"""

# rst IMPORTS start
from prefoil import Airfoil
from prefoil.utils import readCoordFile, sampling
import matplotlib.pyplot as plt

# rst IMPORTS end

"""
Here we read an airfoil coordinate file from a database, perform geometric
cleanup, and then sample it with a particular distribution.
"""

# rst PLOT start
# Read the Coordinate file
filename = "../tests/airfoils/rae2822.dat"
coords = readCoordFile(filename)
airfoil = Airfoil(coords)

# Plot the airfoil
fig1 = airfoil.plot()
fig1.suptitle("Test single distribution")
plt.show()
# rst PLOT end

# # -------------------------------------------------------------------
"""
Here we sample the airfoil with two different distributions, and examine them side
by side.
"""

# rst COMPSAMPLE start
# Compare two different conical coefficient samplings
coords = airfoil.getSampledPts(50, spacingFunc=sampling.conical, func_args={"coeff": 1})
fig2 = airfoil.plot()
fig2.suptitle("coeff = 1")
coords2 = airfoil.getSampledPts(50, spacingFunc=sampling.conical, func_args={"coeff": 3})
fig3 = airfoil.plot()
fig3.suptitle("coeff = 3")
plt.show()
# rst COMPSAMPLE end

# -----------------------------------------
"""
Here we sample the airfoil with two separate distributions for the upper and
lower surfaces. We also thicken the trailing edge, and specify npts_TE during
sampling. These points are then saved to a plot3d file for use with pyHyp.
"""

# rst ULSAMPLING start
# Sample with a bigeometric on the upper surface and eigth order polynomial on the lower surface
coords = airfoil.getSampledPts(
    50, spacingFunc=[sampling.bigeometric, sampling.polynomial], func_args=[{}, {"order": 8}]
)

# Plot the sample
fig4 = airfoil.plot()
fig4.suptitle("Test double distribution")
plt.show()
# rst ULSAMPLING end

# ---------------------------------------------
"""
Here we write out the previous sampling to a plot3d surface mesh that pyhyp
can use and generate an FFD. This sets us up to use the rest of the mach-aero
framework to run an optimization.
"""

# rst OPTSETUP start
# Write surface mesh
airfoil.writeCoords("rae2822", file_format="plot3d")

# Write a fitted FFD with 10 chordwise points
airfoil.generateFFD(10, "ffd")
# rst OPTSETUP end
