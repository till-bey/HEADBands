# One frisbee candidate in an image (full-resolution pixels, origin top-left).
float64 u                 # ellipse center
float64 v
float64 bbox_x            # bounding box, top-left corner and size
float64 bbox_y
float64 bbox_w
float64 bbox_h
float64 major_px          # fitted ellipse axes; the major axis is the apparent diameter
float64 minor_px          # (tilt shrinks only the minor axis)
float64 angle_deg
float32 confidence        # 0..1 shape score (heuristic, not a calibrated probability)
