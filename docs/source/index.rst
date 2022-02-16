.. track2trajectory documentation master file, created by
   sphinx-quickstart on Wed Feb 16 18:38:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

track2trajectory: Matching 2D tracks to 3D trajectories
=======================================================

This is a work under progress...a skeleton online docs page.

What this package does
----------------------
Have you recorded multiple animals moving around in groups on two cameras? You managed to track each animal 
in 2D across the two cameras, but are now wondering which tracks correspond to each other in 3D? :code:`track2trajectory` 
solves this problem for you. 

Assumptions
-----------
This package assumes:
	* You have uniquely identified 2D tracks of objects captured across two (and only two) cameras
	* The two cameras are temporally synchronised (frame N of camera 1 captures the same objects as frame N of camera 2)
	* All 2D tracks (and their pixel coordinates) are `undistorted`. This is a `very` important point to remember.
	* Your cameras are calibrated. This means you know their intrinsic (focal length, principal points) and extrinsic parameters
	(positions, pose)


Quick example
-------------
	







Conception and history
----------------------
The original code and research for this project was written by Giray Tandogan as part of a Bachelor's Thesis at the University of Konstanz. The project was supervised by Hemal Naik, and Thejasvi Beleyur. Thejasvi Beleyur is currently working
on refactoring the code to make it into an installable package of general interest.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
