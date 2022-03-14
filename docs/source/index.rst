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
in 2D in each camera, but are now wondering which tracks correspond to the same animal in 3D? :code:`track2trajectory` 
solves this problem for you. 

Assumptions
-----------

This package assumes:

	* You have uniquely identified 2D tracks of objects captured across two (and only two!) cameras
	* The two cameras are temporally synchronised (frame N of camera 1 captures the same objects as frame N of camera 2)
	* All 2D tracks (and their pixel coordinates) are `undistorted`. This is a `very` important point to remember.
	* Your cameras are calibrated. This means you know their intrinsic (focal length, principal points) and extrinsic parameters (positions, pose)
	* Your images have `square` pixels


Quick example
-------------
Let's take a look at a quick mock example:

.. code-block::

   from track2trajectory.match2dto3d import estimate_3d_points
   from track2trajectory.camera import Camera

   # Load 2D tracks data across both cameras
   bothcamera_2d_tracks = ...

   # Initialise cameras by specifying their intrinsic and extrinsic parameters
   cam1 = Camera(......)
   cam2 = Camera(......)

   # having loaded the xy data for both cameras now begin the matching
   matched_trajectories = estimate_3d_points(cam1, cam2, bothcamera_2d_tracks, ...)


.. toctree::
      :maxdepth: 4
      :caption: Use Cases

      gallery_examples/index.rst


Conception and history
----------------------
The original code and research for this project was written by Giray Tandogan as part of a Masters Thesis at the University of Konstanz. The project was supervised by Hemal Naik, and Thejasvi Beleyur. Thejasvi Beleyur is currently working on refactoring Giray's code to make it into an installable package of general interest.

Funding
-------
TB was funded during the writing of this package by a Medium grant awarded by the Centre for the Advanced Study of Collective Behaviour, University of Konstanz. 

Code docs
=========

.. toctree::
   :maxdepth:1
   :caption: API reference:

   API_track2trajectory.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
