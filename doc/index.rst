.. preFoil documentation master file, created by
   sphinx-quickstart on Fri Oct 16 18:14:31 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======
preFoil
=======

Introduction
============

preFoil is a pySpline-based utility module that allows to flexibly handle airfoil geometries and enable rapid, custom surface mesh generation.

The user can:

   - Initialize the airfoil from a given input file (e.g. from AirfoilTools) and obtain a smoothly-interpolated shape with an arbitrary number of points
   - Modify some of the airfoil characteristics, such as the trailing edge type
   - Compute some geometric parameters (e.g. thickness, camber)
   - Sample the point distribution along the airfoil with a range of built-in specific distributions
   - Output the files into different formats, for visualization purposes or further meshing with :doc:`pyHyp <pyhyp:index>`
   - Generate FFD boxes to preform airfoil optimizations


.. toctree::
   :maxdepth: 1

   install
   airfoil
   utils
   examples
