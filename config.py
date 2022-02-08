error_code = None
visualizations = False
how_many_objects_to_create = 20
force_calc = False
create_objects = False
do_estimations = True
do_kalman_filter_predictions = True
gt_construction = False
multi_run = False # will write onto an existing result file
reverse_kf = False # requires an existing result file!
kf_distance_threshold = 0.3  # meters.
# Amount of data required from previous frames for kalman forecast to be evaluated as a threshold.
kf_frame_required = 5
