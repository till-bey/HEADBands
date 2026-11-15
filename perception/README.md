# 🥏 Part 1: Ground Frisbee Finder

A small **Streamlit website** to detect **white, roughly circular / elliptical frisbee-like shapes on the ground** with classical OpenCV. **No dataset, training, API key or GPU required.** It can run on a CPU.

## Start the website

```bash
uv run streamlit run website/app.py   # from perception/; uv creates .venv from pyproject.toml
```

Open the address printed in the terminal (normally `http://localhost:8501`).

All website inputs, live camera frames, webcam/video CLI frames, and notebook inputs are resized before detection to **640×480 landscape** or **480×640 portrait**. Other aspect ratios receive black padding to preserve shapes. Displayed coordinates, shape thresholds, annotations, and downloads use this processing resolution. Original files are unchanged.

The detector first thresholds the resized image using two conditions, both required:

- **R + G + B ≥ minimum sum** (0–765; default 435).
- **max(R,G,B) − min(R,G,B) ≤ maximum channel difference** (0–255; default 30). This bounds every pairwise channel difference.

It excludes the region above the ground ROI, cleans the binary mask with a 3×3 opening and 5×5 closing, then finds mask contours and fits ellipses. Geometry, size, aspect ratio, and white fill checks reject unsuitable candidates. There is no edge-detection stage.

The app supports local `Dataset/Images` and `Dataset/Videos` folders (including subfolders), included synthetic samples, image/video uploads, camera snapshots, and precise frame-by-frame video navigation. It displays original/annotated images, raw/cleaned white masks, a heuristic match score, and result downloads.

### Run continuous local webcam / video detection

```bash
python webcam.py --input 0
python webcam.py --input my_video.mp4 --output annotated.mp4
# Headless processing:
python webcam.py --input my_video.mp4 --output annotated.mp4 --no-display
```

The Streamlit website itself shows a **snapshot or a selected video frame**, not a live browser stream. `webcam.py` is the continuous video option.

## Suggested tuning

1. **Ground ROI starts at:** for a fixed camera, move the slider to your field horizon; everything above is excluded.
2. **Minimum R + G + B:** lower to include white discs in shadow; raise to reject darker neutral objects. Default 435.
3. **Maximum channel difference:** lower to reject red/pink and other tinted objects; increase cautiously for white discs under warm/cool lighting. Default 30.
4. **Shape:** increase minimum aspect ratio to filter long bright lines; reduce it for edge-on discs.
5. **Radii:** adjust for expected frisbee sizes in the resized images.

The debug views show the **RGB threshold mask** before cleanup and the **cleaned detection mask** used to fit ellipses. The notebook compares channel-difference limits.

## Limitations / interpreting the results

- This is **not a trained image classifier**. It outputs *white, frisbee-like candidates*, not definitive frisbee labels.
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
├── webcam.py           # Continuous webcam / video inference
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
