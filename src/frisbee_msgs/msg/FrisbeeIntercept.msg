# Predicted interception of the frisbee with the catching plane.
# header.stamp = stamp of the newest measurement used; header.frame_id = frame of catch_point.
std_msgs/Header header
bool valid                # false: do not use (see status)
string status             # why it is invalid, or "ok"
string mode               # metric (calibrated camera + TF) | image (pixel calibration)
geometry_msgs/Point catch_point
builtin_interfaces/Time intercept_time
float64 time_to_intercept       # s from header.stamp
float64 position_sigma          # m (metric mode), NaN otherwise
float64 time_sigma              # s
float32 confidence              # 0..1 heuristic
float64 pixel_u                 # predicted image position at interception
float64 pixel_v
bool in_workspace               # catch_point inside the configured workspace box
bool interception_enabled       # automatic interception armed (catch targets are being sent)
