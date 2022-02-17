'''
Prediction module
=================
Contains all the various types of prediction algorithms. Currently only the 
Kalman filter is implemented.

'''

import numpy as np
import pandas as pd
import cv2

def kalman_predict(obj_id, frame_number, prev_matched_data, min_frames=5,
                                               reverse_flag=False):
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
    prev_matched_data : pd.DataFrame
        Dataframe with at least the following columns:
            frame, objectid_cam1, objectid_cam2, x, y, z
    min_frames : int > 0, optional 
        Minimum number of frames that are required to run the prediction. 
        Defaults to 5 frames (or 10????) CLARIFY HERE!!!
    reverse_flag :  bool, optional
        Set to True only if there is already a result file.
        Defaults to False

    Returns
    -------
    4 outputs in the order: x,y,z, number of frames used to predict
    If output is -1 then it means there is not sufficient data.

    TODO
    ----
    The Kalman filter parameters need to be tweaked - or also set to a variable 
    instead of the currently hard-coded values for each of the cv2.KalmanFilter 
    objects
    '''
    # For 3D
    df_t = prev_matched_data.loc[(prev_matched_data['objectid_cam1'].astype(float) == float(obj_id))]

    kalman_x = cv2.KalmanFilter(2, 1, 0)
    kalman_y = cv2.KalmanFilter(2, 1, 0)
    kalman_z = cv2.KalmanFilter(2, 1, 0)
    if not df_t.empty:
        state_value_x = np.array([[0.0], [df_t.iloc[0]['x']]])
        state_value_y = np.array([[0.0], [df_t.iloc[0]['y']]])
        state_value_z = np.array([[0.0], [df_t.iloc[0]['z']]])


        kalman_x.transitionMatrix = np.array([[1., 1.], [0., 1.]])  # F. input
        kalman_x.measurementMatrix = 1. * np.eye(1, 2)  # H. input
        kalman_x.processNoiseCov = 100 * np.eye(2)  # Q. input
        kalman_x.measurementNoiseCov = 5 ** 2 * np.ones((1, 1))  # R. input
        kalman_x.errorCovPost = 5 ** 2 * np.eye(2, 2)  # P._k|k  KF state var
        kalman_x.statePost = 0.1 * np.random.randn(2, 1)
        kalman_x.statePost = state_value_x  # x^_k|k  KF state var


        kalman_y.transitionMatrix = np.array([[1., 1.], [0., 1.]])  # F. input
        kalman_y.measurementMatrix = 1. * np.eye(1, 2)  # H. input
        kalman_y.processNoiseCov = 100 * np.eye(2)  # Q. input
        kalman_y.measurementNoiseCov = 5 ** 2 * np.ones((1, 1))  # R. input
        kalman_y.errorCovPost = 5 ** 2 * np.eye(2, 2)  # P._k|k  KF state var
        kalman_y.statePost = 0.1 * np.random.randn(2, 1)
        kalman_y.statePost = state_value_y  # x^_k|k  KF state var


        kalman_z.transitionMatrix = np.array([[1., 1.], [0., 1.]])  # F. input
        kalman_z.measurementMatrix = 1. * np.eye(1, 2)  # H. input
        kalman_z.processNoiseCov = 100 * np.eye(2)  # Q. input
        kalman_z.measurementNoiseCov = 5 ** 2 * np.ones((1, 1))  # R. input
        kalman_z.errorCovPost = 5 ** 2 * np.eye(2, 2)  # P._k|k  KF state var
        kalman_z.statePost = 0.1 * np.random.randn(2, 1)
        kalman_z.statePost = state_value_z  # x^_k|k  KF state var

        counter = frame_number - min_frames*2
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
                if cal_counter >= min_frames:
                    mt = np.squeeze(np.asarray(v1))
                    if counter == frame_number - 1:
                        if mt.size > 1:
                            return point_fc_x[0][0], point_fc_y[0][0], point_fc_z[0][0], cal_counter

                if counter == frame_number - 1:
                    # print("Error at kalman_predict()")
                    return -1, -1, -1, cal_counter

    else:
        return -1, -1, -1, -1