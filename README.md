# tracks2trajectory

```tracks2trajectory``` allows automated correspondence of 2D object trajectories into 3D trajectories.


## Before you start
To begin a matching run you need the following inputs to start with. 

1. Camera parameters : the intrinsic and extrinsic parameters of the camera must be known. 
1. 2D tracks of particles of two cameras: Currently only TWO camera matching is supported. The data needs to have the following format:

| frame | object_id | camera_id  | x | y |
|-------|-----|------|---|---|
|   0   | 1   |   0  | 3 | 8 |
|   1   | 1   |   0  | 4 | 9 |
|   0   | 2   |   1  | 6 | 10|


where ```frame``` is the frame number, ```object_id``` is object id, ```camera_id``` is camera id, ```x``` is the pixel column coordinate (increases left to right), ```y``` is the pixel row coordinate (increases top to bottom). 

The  2D tracks across the two cameras need to be loaded into a ```pd.DataFrame``` object first.

## A short example

```
import tracks2trajectory as t2t

# load the 2D tracking data
object_tracks = pd.read_csv('yourfilenamehere.csv')

# begin the 2D to 3D trajectory matching 

matched_3D_trajectories = t2t.match_2D_to_3D(object_tracks)

```

The final output ```matched_3D_trajectories``` is a ```pd.DataFrame``` with the following columns: ```frame```, ```objectid_cam1```, ```objectid_cam2```, ```x```, ```y```, ```z```, ```dist_to_cam1_epipolar```, ```dist_to_cam2_epipolar```

| frame | objectid_cam1 | objectid_cam2 | x | y | z | dist_to_cam1_epipolar |dist_to_cam2_epipolar |
|-------|---------------|---------------|---|---|---|-----------------------|----------------------|
| ||||||||


















## Contributors
This code base is built on Giray Tandogan's Master's thesis co-supervised by Hemal Naik and Thejasvi Beleyur at the Uni. Konstanz. 