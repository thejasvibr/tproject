# -*- coding: utf-8 -*-
"""
Testing overview documents
==========================
Discussion with Giray on potential tests for functions

tests for find_candidate (rename? to find_closest_to_epipolar_line)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 2 views, choose 1 with 5 possible candidates close to an epipolar line and check.
Cases to test:
    * No points
    * Multiple points with the same distance


Tests for estimate_3d_points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use the synthetic data in synthetic_data with run_setup() on Camera 1 and 2 
and generate the simulated 2D tracking (force_calculation=True, do_estimations=True)

OR generate a separate 2D tracking test data case.


Tests for kalman_predict
~~~~~~~~~~~~~~~~~~~~~~~~
* Generate synthetic data e.g. 0-pi of a sine wave and then ask the KF to predict 
the next time-point

Test calcFundamentalMatrix
~~~~~~~~~~~~~~~~~~~~~~~~~~
No concrete ideas yet. One possibility is when the cameras are at 90degree angles,
and with no distortion and at same height etc. Some kind of analytically tractable case
where the algebra simplifies and one can predict the fundamental matrix? ASk Hemal - he'd know
the ideal cases.

"""

