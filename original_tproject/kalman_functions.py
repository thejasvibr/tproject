'''

Deleted functions in the latest commit:
    
    kalman_fill

'''
import numpy as np
import pandas as pd
from pathlib import Path
import classes
import config
import cv2
import math

def kalman_predict(obj_id, frame_number, file_name, reverse_flag):
    '''
    Checks previous xyz positions of points and uses it to predict the next point.
    Also takes a reverse flag -- which works only if there is a pre-existing result
    file.

    Parameters
    ----------
    obj_id : int
        Object ID
    frame_number : int
        Frame number
    file_name : str
        REsult file name. Expected to be in the 'result_files/' folder. 
    reverse_flag :  bool
        Set to True only if there is already a result file.
        
    Returns
    -------
    4 outputs in the order: x,y,z, number of frames used to predict
    If output is -1 then it means there is not sufficient data.

    TODO
    ----
    The Kalman filter parameters need to be tweaked - or also set to a variable 
    instead of the currently hard-coded values.
    '''
    # print("at kalman_predict")
    path_to_check_1 = Path(f"result_files/{file_name}.csv")
    if path_to_check_1.exists():

        df_matched = pd.read_csv(path_to_check_1)

        # For 2D
        # df_t = df_matched.loc[
        #    (df_matched['oid'].astype(float) == float(obj_id)) & (df_matched['cid'].astype(float) == float(1))]
        # For 3D
        df_t = df_matched.loc[
            (df_matched['oid_1'].astype(float) == float(obj_id))]

        kalman_x = cv2.KalmanFilter(2, 1, 0)
        kalman_y = cv2.KalmanFilter(2, 1, 0)
        kalman_z = cv2.KalmanFilter(2, 1, 0)
        if not df_t.empty:
            state_value_x = np.array([[0.0], [df_t.iloc[0]['x']]])
            state_value_y = np.array([[0.0], [df_t.iloc[0]['y']]])
            state_value_z = np.array([[0.0], [df_t.iloc[0]['z']]])

            state_x = state_value_x  # start state
            kalman_x.transitionMatrix = np.array([[1., 1.], [0., 1.]])  # F. input
            kalman_x.measurementMatrix = 1. * np.eye(1, 2)  # H. input
            kalman_x.processNoiseCov = 100 * np.eye(2)  # Q. input
            kalman_x.measurementNoiseCov = 5 ** 2 * np.ones((1, 1))  # R. input
            kalman_x.errorCovPost = 5 ** 2 * np.eye(2, 2)  # P._k|k  KF state var
            kalman_x.statePost = 0.1 * np.random.randn(2, 1)
            kalman_x.statePost = state_value_x  # x^_k|k  KF state var

            state_y = state_value_y  # start state
            kalman_y.transitionMatrix = np.array([[1., 1.], [0., 1.]])  # F. input
            kalman_y.measurementMatrix = 1. * np.eye(1, 2)  # H. input
            kalman_y.processNoiseCov = 100 * np.eye(2)  # Q. input
            kalman_y.measurementNoiseCov = 5 ** 2 * np.ones((1, 1))  # R. input
            kalman_y.errorCovPost = 5 ** 2 * np.eye(2, 2)  # P._k|k  KF state var
            kalman_y.statePost = 0.1 * np.random.randn(2, 1)
            kalman_y.statePost = state_value_y  # x^_k|k  KF state var

            state_z = state_value_z  # start state
            kalman_z.transitionMatrix = np.array([[1., 1.], [0., 1.]])  # F. input
            kalman_z.measurementMatrix = 1. * np.eye(1, 2)  # H. input
            kalman_z.processNoiseCov = 100 * np.eye(2)  # Q. input
            kalman_z.measurementNoiseCov = 5 ** 2 * np.ones((1, 1))  # R. input
            kalman_z.errorCovPost = 5 ** 2 * np.eye(2, 2)  # P._k|k  KF state var
            kalman_z.statePost = 0.1 * np.random.randn(2, 1)
            kalman_z.statePost = state_value_z  # x^_k|k  KF state var

            counter = frame_number - config.kf_frame_required * 2
            cal_counter = 0

            while counter < frame_number - 1 and len(df_t) > 0:
                # Predict
                prediction_x = kalman_x.predict()
                prediction_y = kalman_y.predict()
                prediction_z = kalman_z.predict()
                if reverse_flag:
                    i_number = frame_number - counter + frame_number
                else:
                    i_number = counter
                # Update
                dff = df_t.loc[df_t['frame'].astype(int) == int(i_number)]
                if len(dff) > 0:
                    vx = np.array([[dff.iloc[0]['x']]])
                    vy = np.array([[dff.iloc[0]['y']]])
                    vz = np.array([[dff.iloc[0]['z']]])
                    v1 = kalman_x.correct(vx)
                    v2 = kalman_y.correct(vy)
                    v3 = kalman_z.correct(vz)

                    # forecast
                    point_fc_x = np.dot(kalman_x.transitionMatrix, kalman_x.statePost)
                    point_fc_y = np.dot(kalman_y.transitionMatrix, kalman_y.statePost)
                    point_fc_z = np.dot(kalman_z.transitionMatrix, kalman_z.statePost)

                    counter += 1
                    cal_counter += 1
                    if counter == float(frame_number):
                        counter = frame_number - 1

                    mt = np.squeeze(np.asarray(v1))
                    if counter == frame_number - 1:
                        if mt.size > 1 and cal_counter > 6:
                            return point_fc_x[0][0], point_fc_y[0][0], point_fc_z[0][0], cal_counter
                        else:

                            return -1, -1, -1, cal_counter

                else:
                    counter += 1
                    if cal_counter >= config.kf_frame_required:
                        mt = np.squeeze(np.asarray(v1))
                        if counter == frame_number - 1:
                            if mt.size > 1:
                                return point_fc_x[0][0], point_fc_y[0][0], point_fc_z[0][0], cal_counter

                    if counter == frame_number - 1:
                        # print("Error at kalman_predict()")
                        return -1, -1, -1, cal_counter

        else:
            return -1, -1, -1, -1




    else:
        print("File cannot be found: " + file_name + " at function: kalman_predict")
        return config.error_code, config.error_code



def kalman_fill(file_name, frame_window, how_many_frames):
    '''
    Not used in the final implementation. Could potentially be used to 
    interpolate between empty frames with un-matched points.
    '''
    path_to_check_1 = Path(f"result_files/{file_name}.csv")
    if path_to_check_1.exists():
        df_matched = pd.read_csv(path_to_check_1)

        counter = 0

        unique_ids = df_matched["gtoid"].unique()
        unique_ids.sort()
        print(unique_ids)
        for id in unique_ids:

            object_data = df_matched.loc[df_matched["oid_1"].astype(int) == int(id)]

            # how_many_frames = len(object_data)
            # print(how_many_frames)
            first_frame_num = object_data.iloc[0]["frame"].astype(int)
            # last_frame_num = object_data.iloc[how_many_frames - 1]["frame"].astype(int)
            diff_frame = len(object_data)

            if how_many_frames != diff_frame:
                print("Not complete object with id: " + str(id))
                temp_frame_value = object_data.iloc[0]["frame"].astype(int)
                for z in range(1, frame_window):
                    frame_value = object_data.iloc[z]["frame"].astype(int)
                    if (frame_value - temp_frame_value) != 1:
                        df_matched = df_matched.append(
                            pd.DataFrame(
                                {"frame": temp_frame_value + 1, "gtoid": object_data.iloc[z]['gtoid'],
                                 "oid_1": object_data.iloc[z]['oid_1'],
                                 "oid_2": object_data.iloc[z]['oid_2'],
                                 "x": object_data.iloc[z]['x'], "y": object_data.iloc[z]['y'],
                                 "z": object_data.iloc[z]['z'], "gtx": object_data.iloc[z]['gtx'],
                                 "gty": object_data.iloc[z]['gty'], "gtz": object_data.iloc[z]['gtz'], "ee": 0.0,
                                 "dt1": object_data.iloc[z]['dt1'], "dt2": object_data.iloc[z]['dt2'],
                                 "dt3": object_data.iloc[z]['dt3']}, index=[0]))
                    else:
                        temp_frame_value = frame_value

                # TODO 2: fill rest of the frames
                temp_estimation = config.error_code
                update_counter = 0

                KFx = classes.KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)
                KFy = classes.KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)
                KFz = classes.KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)
                run_len = len(object_data)
                if run_len > 181:
                    start = run_len - 181
                else:
                    start = 0
                temp_frame_value = object_data.iloc[start]["frame"].astype(int) - 1
                for k in range(start, run_len):
                    # Predict
                    (xv, t4) = KFx.predict()
                    (yv, t5) = KFy.predict()
                    (zv, t6) = KFz.predict()

                    frame_value = object_data.iloc[k]["frame"].astype(int)

                    # Update
                    row = object_data.loc[object_data['frame'].astype(float) == float(frame_value)]
                    if len(row) > 0:
                        txv = row.iloc[0]['x'].astype(float)
                        tyv = row.iloc[0]['y'].astype(float)
                        tzv = row.iloc[0]['z'].astype(float)

                        (x1, t1) = KFx.update((txv, tyv))
                        (y1, t2) = KFy.update((tyv, tzv))
                        (z1, t3) = KFz.update((tzv, txv))
                        update_counter += 1
                    if (frame_value - temp_frame_value) != 1:
                        # TODO: Assign estimated values and calculate estimation error for estimated value.
                        missing_frames = frame_value - temp_frame_value

                        for t in range(1, missing_frames - 1):
                            print("xv:" + str(xv) + ", gtxv: " + str(row.iloc[0]['x']) + ", update_counter: " +
                                  str(update_counter) + " frame: " + str(temp_frame_value + t))
                            mt1 = np.squeeze(np.asarray(xv))
                            mt2 = np.squeeze(np.asarray(yv))
                            mt3 = np.squeeze(np.asarray(zv))
                            mt1 = mt1.tolist()
                            mt2 = mt2.tolist()
                            mt3 = mt3.tolist()
                            if len(mt1) > 1 and len(mt2) > 1 and len(mt3) > 1:
                                mt2 = mt2[0]
                                mt1 = mt1[0]
                                mt3 = mt3[0]
                                df_matched = df_matched.append(
                                    pd.DataFrame(
                                        {"frame": temp_frame_value + t, "gtoid": object_data.iloc[k]['gtoid'],
                                         "oid_1": object_data.iloc[k]['oid_1'],
                                         "oid_2": object_data.iloc[k]['oid_2'],
                                         "x": mt1, "y": mt2,
                                         "z": mt3, "gtx": object_data.iloc[k]['gtx'],
                                         "gty": object_data.iloc[k]['gty'], "gtz": object_data.iloc[k]['gtz'],
                                         "ee": 0.0,
                                         "dt1": object_data.iloc[k]['dt1'], "dt2": object_data.iloc[k]['dt2'],
                                         "dt3": object_data.iloc[k]['dt3']}, index=[0]))

                                # Predict
                                (xv, t4) = KFx.predict()
                                (yv, t5) = KFy.predict()
                                (zv, t6) = KFz.predict()

                                (x1, t1) = KFx.update((mt1, mt2))
                                (y1, t2) = KFy.update((mt2, mt3))
                                (z1, t3) = KFz.update((mt3, mt1))

                    temp_frame_value = frame_value

            else:
                print("Complete object with id: " + str(id))

        df_matched = df_matched.sort_values(by=['frame', 'gtoid'], ascending=[True, True])
        df_matched.to_csv(f"result_files/recon1.csv", index=False)

