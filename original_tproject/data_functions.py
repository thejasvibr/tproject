import numpy
import numpy as np
import pandas as pd
import random
from pathlib import Path
import cv2
import glob
import config
import classes


def read_starling_data_file(file_name):
    '''
    Reads the custom format of the starling dataset and formats it into the 
    normal groundtruth data.
    
    Parameters
    ----------
    file_name : str
        File name template that is expected to be multiple csv files in the sgt_files folder of 
        the working directory
    
    Returns 
    -------
    csv_to_df : pd.DataFrame 
    True/False : loading success
    
    Side Effects
    ------------
    writes 'result_files/combined_csv.csv' - which has the 3D trajectories of points
    from the starling data. 
    
    '''
    path_to_check = Path(f"sgt_files/{file_name}.csv")

    extension = 'csv'
    all_filenames = [i for i in glob.glob('sgt_files/*.{}'.format(extension))]
    print(all_filenames)
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # combined_csv.to_csv("result_files/combined_csv.csv", index=False, encoding='utf-8-sig')

    if True:
        # csv_to_df_pre = pd.read_csv(path_to_check)
        csv_to_df_pre = combined_csv

        # Convert patternIDs to oids.
        unique_ids = csv_to_df_pre["patternID"].unique()
        starting_id = 0
        for i in unique_ids:
            csv_to_df_pre.loc[csv_to_df_pre["patternID"] == i, "patternID"] = starting_id
            starting_id += 1
        data_to_transfer = csv_to_df_pre[['Frame', 'patternID', 'X', 'Y', 'Z']].copy()
        data_to_transfer = data_to_transfer.rename({'Frame': 'frame', 'patternID': 'oid',
                                                    'X': 'x', 'Y': 'y', 'Z': 'z'}, axis='columns')
        # Convert mm to meters.
        data_to_transfer['x'] = data_to_transfer['x'].astype('float64').div(1000)
        data_to_transfer['y'] = data_to_transfer['y'].astype('float64').div(1000)
        data_to_transfer['z'] = data_to_transfer['z'].astype('float64').div(1000)

        csv_to_df = data_to_transfer
        csv_to_df.to_csv("result_files/combined_csv.csv", index=False, encoding='utf-8-sig')
        return csv_to_df, True
    else:
        print("File cannot be found.")
        return error_code, False


def create_data_3d_object_data_2(how_many, fps, how_long_seconds):
    '''
    Creates 3D trajectories for later 2D projection. Brownian-type motion 
    within a 3D box
    
    Parameters
    ----------
    how_many : int >0 
        NUmber of synthetic particles
    fps : int >0 
        Frames per second
    how_long_seconds : float>0
        Duration of the synthetic 'recording'

    Side effects
    ------------
    Outputs all the  groundtruth (gt) 3D trajectories to 'gt_files/gt_3d_data_v1.csv'
    The dataframe has 
        frame
        oid : object id
        x,y,z : position 
    '''
    object_3d_df = pd.DataFrame(
        columns=[
            "frame", "oid", "x", "y", "z"
        ])

    object_starting_x, object_starting_y, object_starting_z = 1, 0, 2.5

    for i in range(how_many):
        t_oid = i
        t_x = object_starting_x + random.randint(0, 1) / 10
        t_y = object_starting_y + random.randint(0, 1) / 10
        t_z = object_starting_z + random.randint(0, 1)
        o_x = random.randint(0, 10)
        o_y = random.randint(0, 10)
        o_z = random.randint(0, 10)
        for k in range(fps * how_long_seconds):
            sv = np.sin(k)

            if o_x > 5:
                t_x += (sv + random.randint(0, 14)) / fps / 3
            else:
                t_x -= (sv + random.randint(0, 14)) / fps / 3

            if o_y > 5:
                t_y += (sv + random.randint(0, 14)) / fps / 3
            else:
                t_y -= (sv + random.randint(0, 14)) / fps / 3

            if o_z > 5:
                t_z += (sv + random.randint(0, 14)) / fps / 3
            else:
                t_z -= (sv + random.randint(0, 14)) / fps / 3

            # For cave-ish scenario
            # if t_x > 5:
            #     o_x = 0
            # elif t_x < -5:
            #     o_x = 10
            # if t_y > 5:
            #     o_y = 0
            # elif t_y < 0.0:
            #     o_y = 10
            # if t_z > 5:
            #     o_z = 0
            # elif t_z < 1:
            #     o_z = 10

            if t_x > 2.5:
                o_x = 0
            elif t_x < -0.5:
                o_x = 10
            if t_y > 0.6:
                o_y = 0
            elif t_y < -0.6:
                o_y = 10
            if t_z > 3:
                o_z = 0
            elif t_z < 2.5:
                o_z = 10

            object_3d_df = object_3d_df.append(pd.DataFrame({"frame": [k], "oid": [t_oid],
                                                             "x": [t_x], "y": [t_y], "z": [t_z]}))
    object_3d_df.to_csv("gt_files/gt_3d_data_v1.csv", index=False)


def gt_data_to_csv(df_to_csv, what_to_do, file_name):
    # df_to_csv = pd.DataFrame(columns=["frame", "oid", "x", "y", "z"])

    path_to_check = Path(f"gt_files/{file_name}.csv")

    if what_to_do == "create":

        df_to_csv.to_csv(path_to_check, index=False)

        return 1

    elif what_to_do == "append" and path_to_check.exists():
        read_df = pd.read_csv(path_to_check)
        read_def_last_oid = read_df['oid'].max()
        oid_counter = read_def_last_oid
        oid_counter += 1  # "oid_counter + 1" in order to continue from last object

        df_to_csv = df_to_csv

        df_to_csv.to_csv(path_to_check, mode='a', header=False, index=False)
        # print(df_to_csv)
        return 1
    else:
        print("Something went wrong at gt_data_to_csv().")
        return config.error_code


def read_data_file(file_name):
    '''
    Parameters
    ----------
    file_name : str,path
        expects the file to be in 'gt_files' folder within the working 
        directory
    Returns
    -------
    csv_to_df : pd.DataFrame
    True/False : success indicator
    '''
    path_to_check = Path(f"gt_files/{file_name}.csv")

    if path_to_check.exists():
        csv_to_df = pd.read_csv(path_to_check)
        return csv_to_df, True
    else:
        print("File cannot be found.")
        return config.error_code, False


def project_to_2d_and_3d(gt_3d_df, camera_obj: classes.Camera, mu, sigma, mu3d, sigma3d):
    '''
    Projects 2D points from the input 3D trajectories while also adding noise. 
    Two types of noise are implemented (both can be also used). Pre-projection noise
    adds 3D noise to the xyz coordinates, and post-projection nosie adds 2D noise 
    to the pixel coordinates. 
    
    The projection is always expected to be within 1920 x 1080 pixels
    
    Parameters
    ----------
    gt_3d_df : pd.DataFrame 
        With 3D trajectory information and columns frame, oid, cid and x,y,z
    camera_obj : classes.Camera instance
    mu : float
        2D noise mean (noise post projection)
    sigma : float
        2D noise standard deviation (post projection)
    mu3d : float
        3D noise mean pre-projection 
    sigma3D : float
        3D noise standard deviation pre-projection
    
    Returns
    -------
    proj_2d_df : pd.DataFrame
        2D projection dataframe with columns:
            frame
            oid (object id)
            cid (camera id)
            x : column number (increases to the right)
            y : the row number (increases as you move down the image)

    fail_counter : int
        Number of positions that can't be projected onto the 2D image.

    Side effect
    -----------
    Writes the 2D projections as 'gt_files/proj_2d.csv'
    
    Attention
    ---------
    if a 2D projection file is already there (as gt_files/proj_2d.csv), then 
    unless force_calculation is True - the old file is used. 
    
    '''
    proj_2d_df = pd.DataFrame(columns=["frame", "oid", "cid", "x", "y"])
    path_to_check_1 = Path("gt_files/proj_2d.csv")
    oid = -1
    fail_counter = 0
    if path_to_check_1.exists() and not config.force_calc:
        proj_2d_df = pd.read_csv(path_to_check_1)
    else:
        for i in range(len(gt_3d_df)):
            # 3D noise
            noise3d = np.random.normal(mu3d, sigma3d, [3, 1])
            # Set values & add noise. Divided by 60 due to distribution to achieve up to 10cm+ noise.
            x_temp = gt_3d_df.iloc[i]['x'] + noise3d[0] / 120.0
            y_temp = gt_3d_df.iloc[i]['y'] + noise3d[1] / 120.0
            z_temp = gt_3d_df.iloc[i]['z'] + noise3d[2] / 120.0
            frame_temp = gt_3d_df.iloc[i]['frame']

            oid = gt_3d_df.iloc[i]['oid']
            cid = camera_obj.id
            obj_xyz_temp = [x_temp, y_temp, z_temp]
            obj_xyz_temp = numpy.float32(obj_xyz_temp)

            # Project Points from 3D to 2D
            r_m, _ = cv2.Rodrigues(camera_obj.r_mtrx)
            oi, _ = cv2.projectPoints(obj_xyz_temp, r_m,
                                      camera_obj.t_mtrx, camera_obj.i_mtrx, camera_obj.cof_mtrx)

            # Check if object is in image plane
            if 1920.0 >= oi[0][0][0] >= 0.0 and 0.0 <= oi[0][0][1] <= 1080.0:
                # Set 2D points to data frame.
                # Gaussian noise
                noise = np.random.normal(mu, sigma, [2, 1])
                x2d = np.float32(noise[0] + oi[0][0][0])
                y2d = np.float32(noise[1] + oi[0][0][1])
                proj_2d_df = proj_2d_df.append(pd.DataFrame({"frame": [frame_temp], "oid": [oid],
                                                             "x": x2d, "y": y2d, "cid": cid}))

            else:
                # print("Point cannot be projected: " + str(oi[0][0][0]) + ", " + str(oi[0][0][1]) + " Frame: " + str(i))
                # print("Camera id: " + str(camera_obj.id))
                fail_counter += 1

        # del x_temp, y_temp, z_temp, frame_temp, oid, obj_xyz_temp, oi, \
        #    proj_2d, projected_3d_x, projected_3d_y, projected_3d_z

        # Write dataframe to csv file.
        if path_to_check_1.exists() and camera_obj.id != 0:
            proj_2d_df.to_csv(path_to_check_1, mode='a', header=False, index=False)
            print("appending " + str(oid))
        else:
            proj_2d_df.to_csv(path_to_check_1, index=False)
            print("writing " + str(oid))
    print("Failed total points: " + str(fail_counter))
    return proj_2d_df, fail_counter