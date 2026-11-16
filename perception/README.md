# 🥏 Part 1: Ground Frisbee Finder

A small **Streamlit website** to detect **white circular / elliptical frisbee-like regions** with classical OpenCV and measure their interior brightness. **No dataset, training, API key or GPU required.** It can run on a CPU.

## Start the website

```bash
# From the repository root:
uv run --project perception streamlit run perception/website/app.py

# Or from perception/:
uv run streamlit run website/app.py
```

Open the address printed in the terminal (normally `http://localhost:8501`).
Imports and dataset paths are resolved relative to the app, independent of the
terminal's working directory. `uv` creates the environment from `perception/pyproject.toml`.

### Use the navigation dashboard's saved configuration

On startup, Streamlit reads `src/base_navigation/settings.json`, using the same
validation and legacy defaults as the navigation dashboard. It initializes the
detector resolution, RGB thresholds, ellipse limits, camera URL
and stream format, mounting height/tilt offsets, base height, and FOV from that file.
**Reload saved configuration** in the sidebar picks up changes saved from the
other dashboard. Streamlit controls remain available for temporary tuning and do
not write back to the controller's configuration.

Under **Configuration source**, select a different settings file or navigation
dashboard URL if needed. If the default file belongs to another OS user, Streamlit
can read its saved contents through the running dashboard's read-only
`/settings/saved` endpoint (default `http://127.0.0.1:8765`). Restart the navigation
dashboard after updating this code to expose that endpoint. Read failures produce
a visible warning; a failed reload keeps the last successfully loaded values.
Reconnect cameras after reloading a changed URL or format. Live ROS body/head
feedback is not stored in the settings file; ground-position controls show the
saved mounting calibration, which can be adjusted for offline visualization.

Website inputs, live camera frames, and webcam/video CLI frames use a selectable processing resolution: **320×240, 480×360, 640×480 (default), 960×720, or 1280×960** in landscape, with dimensions swapped for portrait. Notebook inputs keep the 640×480 default unless a width is passed to `prepare_frame`. Other aspect ratios receive black padding to preserve shapes. Displayed coordinates, shape thresholds, annotations, and downloads use this processing resolution. Original files are unchanged.

### Processing speed

Use **Processing resolution** in the sidebar to reduce the image size. Try **320×240** first: it has one quarter as many pixels as 640×480, but may miss distant discs. Radius and area thresholds are in processing pixels and are not scaled automatically; halve radius limits and quarter area limits when halving the resolution. Narrow the radius range to the sizes you expect.

The detector uses RGB white thresholding, mask cleanup, and contour ellipse
fitting. Radius, area, aspect ratio, ellipse fit, white fill, and the surrounding
white ring checks determine accepted candidates. The original fit tolerance,
scoring, and overlap suppression are preserved.

**Mean R+G+B (0–765)** and **Std dev R+G+B** are shown in the candidate table and
image annotations for images, selected video frames, and live camera feeds.
They are the mean and population standard deviation of each pixel's R+G+B
within an inner ellipse whose two radii are 65% of the fitted
ellipse's radii (42.25% of its geometric area), excluding the outer rim. Dark
markings and non-white pixels in this inner region are included. It is measured
before annotations, rather than using only white-threshold pixels.
The live-frame cache retains the measurements alongside the detections.

**Side** is reported in the candidate table and image annotations using the
unrounded R+G+B population standard deviation: **0** below the threshold, **1**
at or above it. Adjust **Side standard deviation threshold** in Detection
settings (default: 30). Save it in the navigation dashboard and reload the saved
configuration here to retain it across sessions. This landing-face label does
not change candidate acceptance or scoring.

RGB preprocessing uses OpenCV channel operations, and ellipse validation scans candidate patches instead of full-image masks. The live Detection dashboard reuses results for unchanged frames/settings. Its UI still refreshes up to about 7 times per second, independently of detector processing speed.

From the `perception` directory:

```bash
python -m camera.webcam --input 0 --resolution 320
python -m camera.webcam --input my_video.mp4 --resolution 480 --min-radius 6 --max-radius 60
python -m camera 'http://192.168.1.50:8080/stream.mjpg' --resolution 320
```

For the camera API and ground-position dashboard, the same processing resolution applies. The camera API returns coordinates in the full-resolution upright portrait frame, undoing detector resizing and padding:

```python
settings = DetectorSettings(processing_width=320,
                            min_radius=5, max_radius=60, min_area=40)
# Use settings with FrisbeeCamera(url, settings).
# Direct frame preprocessing: prepare_frame(frame, processing_width=320).
```

The detector thresholds the resized image using two conditions, both required:

- **R + G + B ≥ minimum sum** (0–765; default 160).
- **max(R,G,B) − min(R,G,B) ≤ maximum channel difference** (0–255; default 55). This bounds every pairwise channel difference.

The detector cleans the binary mask with a 3×3 opening and 5×5 closing before
fitting ellipses to the mask's outer contours. The debug views show the RGB
threshold mask and cleaned white mask. Image edges are not a candidate source.
The RGB average does not affect detection or determine which face landed upward.

The app supports local `Dataset/Images` and `Dataset/Videos` folders (including subfolders), included synthetic samples, image/video uploads, camera snapshots, and precise frame-by-frame video navigation. It displays original/annotated images, raw/cleaned white masks, a heuristic match score, and result downloads.

### Run continuous local webcam / video detection

```bash
python -m camera.webcam --input 0
python -m camera.webcam --input my_video.mp4 --output annotated.mp4
# Headless processing:
python -m camera.webcam --input my_video.mp4 --output annotated.mp4 --no-display
```

The Streamlit website itself shows a **snapshot or a selected video frame**, not a live browser stream. `camera/webcam.py` is the continuous video option.

## Camera center-position class

`camera/frisbee_camera.py` provides `FrisbeeCamera`, using the same white RGB thresholds, mask cleanup, and ellipse fitting as the website. From the `perception` directory:

```python
from camera import FrisbeeCamera
from detector import DetectorSettings

# Set these to match your website sliders; slider values are not saved automatically.
settings = DetectorSettings(
    rgb_sum_min=160,
    channel_difference_max=55,
    min_white_fraction=0.48,
)

with FrisbeeCamera("http://192.168.1.50:8080/stream.mjpg", settings) as camera:
    center = camera.get_center(timeout=5.0)
    if center is not None:
        x, y = center
        print(f"Frisbee center: x={x}, y={y}")
    else:
        print("No frisbee found")
```

Live-camera coordinates are integer pixels in the **upright portrait frame**, measured from the top-left corner: X increases rightward, Y downward. Landscape-shaped captured frames rotate 90° clockwise before detection and projection; already-portrait frames stay unchanged. The detector still processes resized/padded images just like the website. With several candidates, it returns the highest-scoring white-disc candidate.

Call `get_center()` repeatedly inside the `with` block for continuous use. Each call waits for a new frame; missing or stale frames raise `TimeoutError`, while a valid frame without a detection returns `None`. The timeout covers frame acquisition, not detection time. The context manager closes the camera reader; without it, call `camera.close()` yourself. For an HTTP JPEG snapshot URL, pass `mode=SNAPSHOT` imported from `camera`. RTSP and MJPEG URLs use the default mode.

You can also detect a center in an existing OpenCV BGR image with `camera.detect_center(frame)` without starting a camera connection; explicit frames keep their supplied orientation. `camera.read_frame(timeout=5.0)` returns a fresh, full-resolution upright portrait BGR frame for display or analysis; it shares the same freshness and timeout rules and consumes that frame for subsequent camera calls.

For a single result from the terminal:

```bash
python -m camera 'http://192.168.1.50:8080/stream.mjpg'
# HTTP snapshot endpoint:
python -m camera 'http://192.168.1.50/snapshot.jpg' --snapshot --timeout 10
```

The script prints JSON such as `{"center": [400, 320]}`, or `{"center": null}` when no frisbee is found. Connection timeouts exit with an error. Use the Python class to customize thresholds; the command-line script uses the defaults.

### Website ground-position tab

Open the **Ground position** tab next to **Detection**. Enter a direct camera URL and stream format, then click **Connect ground-position camera**. This dashboard uses the selected processing resolution and white-disc method. It imports `FrisbeeCamera` and `pixel_to_ground` from `camera/frisbee_camera.py` and uses the **White discs** sidebar settings.

Set robot vertical lift Z in meters above the fixed **0.83 m base height**, downward camera tilt in **radians**, horizontal FOV, and optionally vertical FOV (FOV inputs remain degrees). Camera height is calculated as **0.83 + Z meters**; robot ground X/Y are fixed at zero. Tilt is zero horizontal and positive downward. The dashboard updates the upright portrait image with its detected center and shows:

- Camera pixel X/Y in the full-resolution upright image.
- Camera tilt and the detected ray's depression angle below the ground horizon, in radians.
- Final ground bearing in radians (0 forward, positive right), distance, and ground X/Y in meters.

Live camera acquisition defaults to **portrait clockwise**: landscape-shaped frames rotate 90° before detection, while already-portrait frames stay unchanged. Displayed pixel X/Y, the ground projection, and arrow all use the same upright frame at full resolution. The standalone camera API uses this same portrait interpretation.

Set **FOV calibration axes** to **Displayed image** when your FOV values describe the upright portrait frame. If they come from landscape camera specifications, choose **Landscape sensor** to swap the axes for portrait. If vertical sensor FOV is omitted, it is derived from the image aspect ratio and square-pixel assumption before swapping.

An amber **1 m forward** arrow projects the ground segment from directly below the camera to one meter straight ahead. Its tip marks the real one-meter point using the current lift, tilt, and FOV. The tail is clipped if the ground origin is below the frame. If the endpoint is outside the image or behind the camera, a message appears instead. The overlay also appears when no frisbee is detected.

Calibration changes recalculate the position. Missing detections, stale/missing frames, and nonintersecting rays show a status message instead of a ground result. Use **Stop ground-position camera** to close its independent camera connection.

### Robot lift and radians API

For the robot, use the fixed 0.83 m base camera height plus the current vertical lift:

```python
from camera import FrisbeeCamera

with FrisbeeCamera("http://192.168.1.50:8080/stream.mjpg") as camera:
    position = camera.get_robot_ground_position(
        robot_z=0.20,           # lift above base, in meters; camera height = 1.03 m
        tilt_rad=0.261799,      # zero horizontal, positive downward
        horizontal_fov_deg=90,
    )
    print(position)             # x, y, distance, bearing_rad, bearing_deg; or None
```

`detect_robot_ground_position(frame, ...)` accepts the same robot inputs for an existing BGR frame. `pixel_to_ground_robot(...)` projects an independently obtained pixel center with the same base-height offset. Robot ground X/Y stay at zero. The original degree-based APIs below are preserved.

### Real-world ground position

The pixel methods `get_center()` and `detect_center(frame)` and the existing command-line output are unchanged. The same module now includes `pixel_to_ground(...)` plus camera methods that project a detected center onto flat ground:

```python
from camera import FrisbeeCamera

with FrisbeeCamera("http://192.168.1.50:8080/stream.mjpg") as camera:
    position = camera.get_ground_position(
        camera_height=2.0,       # meters (or any consistent distance unit)
        tilt_deg=15,            # downward from horizontal
        horizontal_fov_deg=90,
        # vertical_fov_deg=60,  # optional; omitted means square pixels
        timeout=5.0,
    )
    print(position)
    # {"distance": ..., "bearing_deg": ..., "x": ..., "y": ...}, or None
```

For an existing BGR frame, use `camera.detect_ground_position(frame, camera_height=..., tilt_deg=..., horizontal_fov_deg=..., vertical_fov_deg=...)`. Both methods use the **input frame dimensions and center coordinates**, before detector resizing/padding (upright portrait dimensions for captured camera frames), and return `None` when no frisbee is found. Invalid geometry or a detected ray on/above the horizon raises `ValueError`; camera acquisition still raises `TimeoutError` if no fresh frame arrives.

The ground origin is directly below the camera: X points right and Y forward. `distance` is horizontal ground distance from that origin, in the same units as `camera_height`. `bearing_deg` is zero forward and positive to the right. The model assumes flat ground, no camera roll, an undistorted image, and that the detected frisbee center lies on the ground. FOV values must describe the frame used for projection: the upright portrait axes for captured camera frames. The camera API does not automatically swap FOV values.

To project an independently obtained pixel center, the supplied function is also available directly:

```python
from camera import pixel_to_ground

position = pixel_to_ground(
    px=1100, py=650,
    image_width=1920, image_height=1080,
    camera_height=2.0, tilt_deg=15, horizontal_fov_deg=90,
)
print(position)
```

## Suggested tuning

1. **Minimum R + G + B:** lower to include white discs in shadow; raise to reject darker neutral objects. Default 160.
2. **Maximum channel difference:** lower to reject red/pink and other tinted objects; increase cautiously for white discs under warm/cool lighting. Default 55.
3. **Shape:** increase minimum aspect ratio to filter long bright lines; reduce it for edge-on discs.
4. **Radii:** adjust for expected frisbee sizes in the resized images.

The debug views show the **RGB threshold mask** before cleanup and the **cleaned
detection mask** used for ellipse fitting. The notebook compares channel-difference limits.

## Limitations / interpreting the results

- This is **not a trained image classifier**. It outputs *white elliptical candidates*, not definitive frisbee labels.
- The detection scores are simple heuristics **not calibrated probabilities**.
- White plates, lids, field markings, bright patches, and similarly shaped objects can trigger false positives.
- It does **not** independently establish that an object actually touches the ground.
- The synthetic samples are for functional demos; real-world performance needs testing on your actual camera frames.
- Small distant discs may be missed. Strong color casts can exclude white discs; clipped highlights on colored objects can appear neutral and pass the white mask.

## Project files

```
perception/
├── Dataset/            # Your images and videos (demo files included)
│   ├── Images/
│   └── Videos/
├── website/            # Streamlit website and media browsing
│   ├── __init__.py
│   ├── app.py
│   └── media_utils.py
├── detector/           # RGB thresholds, ellipse fitting, drawing
│   └── __init__.py
├── camera/             # Shared camera API for website and navigation
│   ├── __init__.py
│   ├── __main__.py     # python -m camera
│   ├── camera_stream.py # Background MJPEG/RTSP/snapshot reader
│   ├── frisbee_camera.py # Detection and ground projection
│   ├── target_reader.py # Navigation camera worker and setup
│   └── webcam.py       # Continuous webcam / video inference
├── sample_generator/   # Re-create synthetic samples
│   ├── __init__.py
│   └── make_samples.py
├── ground_frisbee_demo.ipynb # Interactive notebook example
├── samples/            # Four generated test images and sample MP4
├── tests/
├── ML/                 # YOLO notebook and pipeline (uv sync --group ml)
├── ultralytics_dataset/ # YOLO frames, labels, exp.pt
├── pyproject.toml      # deps: uv sync (site), uv sync --group ml (YOLO)
└── README.md
```

To regenerate test examples: `python -m sample_generator.make_samples`. Generated images and the demo MP4 are stored in `samples/`. Select **Sample image** or **Sample video** in the website to view them.

For the notebooks, run `uv sync --group ml`, then `uv run jupyter lab` and open `ground_frisbee_demo.ipynb` or `ML/frisbee_detection.ipynb`.

To run tests: `uv run pytest tests -q`. If a ROS setup is sourced in the shell, run `env -u PYTHONPATH uv run pytest tests -q`: ROS pytest plugins break the venv.

## NEW: Browse local Dataset folders (image files + individual video frames)

Put your own media inside the folders below. These are relative to the **project root**, so they work even if you start Streamlit from somewhere else:

```text
perception/
├── Dataset/
│   ├── Images/
│   │   ├── example_round_white_disc.png      # synthetic demo; replace with your photos
│   │   ├── example_angled_white_disc.png     # synthetic demo
│   │   └── your_photo.jpg
│   └── Videos/
│       ├── example_moving_disc.mp4           # synthetic demo
│       └── your_field_video.mp4
├── website/
│   └── app.py
└── ...
```

In the Streamlit **Input source** selector, choose:

- **Dataset images** to pick a JPG, PNG, BMP, WebP, or TIFF from `Dataset/Images/` (subfolders included).
- **Dataset videos** to pick a video from `Dataset/Videos/` (MP4, MOV, AVI, MKV, M4V, WebM, MPEG; codec support depends on OpenCV).

For videos, drag the **Frame index (0-based)** slider, or click **◀ Previous frame** / **Next frame ▶** to move **one frame at a time**. The page shows the exact frame number, total frames, and approximate timestamp (when the file has FPS metadata). The ellipse detector automatically re-runs on each chosen frame, showing input, annotated image, detection counts, candidate scores, and debug masks.

The video controls use `cv2.VideoCapture` to seek and decode **only the selected frame**, rather than loading the entire video into RAM. Most MP4/H.264 recordings work, but accurate seeking depends on the file and codec. This interface is for frame-by-frame review, not continuous video playback.
