# Image-space track of the frisbee. header.stamp = stamp of the image this update came from.
std_msgs/Header header
uint32 track_id
string state              # tentative | confirmed | coasting | lost
bool confirmed            # true when state is confirmed or coasting
float64 u                 # filtered center, px
float64 v
float64 d                 # filtered apparent diameter, px
float64 du                # px/s
float64 dv
float64 dd                # diameter growth, px/s (positive = approaching)
float64 sigma_uv          # px
uint32 image_width
uint32 image_height
# Recent measurements of this track (oldest first), for trajectory fitting.
float64[] hist_t          # seconds, same clock as header.stamp
float64[] hist_u
float64[] hist_v
float64[] hist_d
float32[] hist_confidence
