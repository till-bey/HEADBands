# Detection changes

Before future changes to detection algorithms, preprocessing, candidate filters,
or candidate scoring/selection, explicitly ask the user to confirm the intended
pipeline change. Do not infer a new detection method from ambiguous terms such
as "frisbee sides": that can mean which face the frisbee landed on.

The authorized pipeline is RGB white thresholding, mask cleanup, and contour
ellipse fitting. Measure the mean and population standard deviation of per-pixel
R+G+B within 65% of both fitted ellipse radii. The user-approved landing-side
rule is side 0 for standard deviation below the configurable side threshold,
side 1 at or above it (default: 30).
These measurements and labels do not affect candidate acceptance or scoring.
Agree on any future landing-side classification rule with the user before
implementing it.
