"""Streamlit UI for ground frisbee detection on folders, video and live IP streams."""
from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from detector import DetectorSettings, detect, draw_detections, prepare_frame
from website.ground_position import render_ground_position
from camera import LatestFrameCamera, STREAM_MODES, VIDEO_STREAM, SNAPSHOT
from website.media_utils import (
    IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, list_media, read_video_frame, video_info,
)

# Media folders live at the project root, one level above the website.
ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "Dataset" / "Images"
VIDEO_DIR = ROOT / "Dataset" / "Videos"
SAMPLE_DIR = ROOT / "samples"
MODEL_NAME = "White discs"

st.set_page_config(page_title="Ground Frisbee Finder", page_icon="🥏", layout="wide")


def image_from_bytes(content: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)


def change_frame(delta: int, frame_count: int) -> None:
    """Streamlit button callback: step exactly one frame, bounded by video length."""
    value = int(st.session_state.get("video_frame_index", 0))
    st.session_state["video_frame_index"] = max(0, min(frame_count - 1, value + delta))


def select_video_frame(video_identity: str, frame_count: int, fps: float) -> int:
    """Share an exact frame-index slider and step buttons across video inputs."""
    if st.session_state.get("selected_video_identity") != video_identity:
        st.session_state["selected_video_identity"] = video_identity
        st.session_state["video_frame_index"] = 0

    if frame_count > 1:
        st.slider(
            "Frame index (0-based)",
            min_value=0,
            max_value=frame_count - 1,
            step=1,
            key="video_frame_index",
            help="Move the slider to any frame, or use the buttons to step one frame at a time.",
        )
    else:
        st.caption("This video contains a single frame.")
        st.session_state["video_frame_index"] = 0
    index = int(st.session_state["video_frame_index"])
    prev, next_, status = st.columns([1, 1, 4], vertical_alignment="center")
    prev.button(
        "◀ Previous frame", on_click=change_frame, args=(-1, frame_count),
        disabled=index == 0, use_container_width=True,
    )
    next_.button(
        "Next frame ▶", on_click=change_frame, args=(1, frame_count),
        disabled=index >= frame_count - 1, use_container_width=True,
    )
    elapsed = f" · {index / fps:.2f} s" if fps > 0 else ""
    status.caption(f"Frame **{index + 1:,} / {frame_count:,}**{elapsed} · exact frame stepping")
    return index


def saved_upload_path(content: bytes, filename: str) -> Path:
    """Save an uploaded video to a stable temporary path for frame seeking."""
    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        suffix = ".mp4"
    digest = hashlib.sha256(content).hexdigest()[:20]
    path = Path(tempfile.gettempdir()) / f"frisbee_ground_uploaded_{digest}{suffix}"
    if not path.exists() or path.stat().st_size != len(content):
        path.write_bytes(content)
    return path


with st.sidebar:
    st.header("Detection settings")
    processing_width = st.selectbox(
        "Processing resolution", [320, 480, 640, 960, 1280], index=2,
        format_func=lambda w: f"{w}×{w * 3 // 4}",
        help="Portrait swaps the dimensions. Lower sizes reduce work but can lose small discs. Radius and area settings use these pixels.",
    )
    st.caption("For speed, start with 320×240.")
    st.subheader("White detection")
    st.caption("Contour ellipses detect round and tilted discs across the full image.")
    rgb_sum_min = st.slider(
        "Minimum R + G + B", 0, 765, 160, 1,
        help="Require R + G + B to reach this value. Lower to include white discs in shadow.",
    )
    channel_difference_max = st.slider(
        "Maximum channel difference", 0, 255, 55, 1,
        help="Require max(R, G, B) − min(R, G, B) to stay below this value. This limits every pairwise channel difference; lower to reject tinted objects.",
    )
    min_white_fraction = st.slider("Minimum white fill fraction", 0.20, 0.95, 0.48, 0.05)
    with st.expander("White size and shape tuning"):
        min_radius = st.slider("Minimum approximate radius (px)", 4, 100, 12)
        max_radius = st.slider("Maximum approximate radius (px)", min_radius + 1, 260, 120)
        min_ellipse_ratio = st.slider("Minimum ellipse aspect ratio", 0.10, 0.95, 0.28, 0.05)
        min_area = st.slider("Minimum white shape area (px²)", 25, 4000, 150, 25)

cfg = DetectorSettings(
    processing_width=processing_width,
    rgb_sum_min=rgb_sum_min,
    channel_difference_max=channel_difference_max,
    min_radius=min_radius,
    max_radius=max_radius,
    min_ellipse_ratio=min_ellipse_ratio,
    min_area=min_area,
    min_white_fraction=min_white_fraction,
)

st.title("🥏 Ground Frisbee Finder")
st.write("Find white frisbee-shaped regions with classical computer vision.")
st.caption(f"Processing at {processing_width}×{processing_width * 3 // 4} landscape, with dimensions swapped for portrait and padding to preserve proportions. Pixel radius/area limits apply at this size.")

detection_tab, position_tab = st.tabs(["Detection", "Ground position"])
# Render the independent dashboard before the existing detector's early stops.
with position_tab:
    render_ground_position(cfg)

with detection_tab:
    source = st.radio(
        "Input source",
        ["Dataset images", "Dataset videos", "Live IP camera", "Sample image", "Sample video", "Upload image", "Upload video", "Camera snapshot"],
        horizontal=True,
    )
    if source != "Live IP camera":
        old_camera = st.session_state.pop("camera_reader", None)
        if old_camera is not None:
            old_camera.stop()
        st.session_state.pop("last_live_result", None)

    frame = None
    frame_description = ""

    if source == "Live IP camera":
        st.subheader("Live network camera")
        st.caption(
            "Connect from the computer running Streamlit, not from the ChatGPT server. "
            "This works with standard HTTP MJPEG and RTSP video endpoints, or with a repeatedly fetched JPEG snapshot URL."
        )
        with st.form("camera_connection_form"):
            camera_url = st.text_input(
                "Direct stream URL (including camera IP, port and path)",
                placeholder="http://192.168.1.50:8080/?action=stream",
                help="Examples: http://IP:8080/?action=stream | http://IP:8080/stream.mjpg | rtsp://IP:8554/cam",
            )
            camera_mode = st.selectbox("Stream format", STREAM_MODES)
            connect = st.form_submit_button("▶ Connect / reconnect", use_container_width=True)
        if connect:
            old_camera = st.session_state.pop("camera_reader", None)
            if old_camera is not None:
                old_camera.stop()
            st.session_state.pop("last_live_result", None)
            try:
                st.session_state["camera_reader"] = LatestFrameCamera(camera_url, camera_mode).start()
                st.success("Connection started. Waiting for the first frame...")
            except ValueError as exc:
                st.error(str(exc))

        camera = st.session_state.get("camera_reader")
        if camera is not None:
            left, right = st.columns([1, 2])
            stop_requested = left.button("■ Stop camera", use_container_width=True)
            right.caption("Automatically reconnects if the stream drops; uses only the newest frame to avoid backlog.")
            if stop_requested:
                camera.stop()
                st.session_state.pop("camera_reader", None)
                st.session_state.pop("last_live_result", None)
                camera = None

        if camera is None:
            st.info("Enter a reachable stream URL and select Connect to begin. Your camera's landing page is usually NOT its direct stream URL.")
        else:
            show_debug = st.checkbox("Show live threshold masks", value=False)
            st.caption("Live dashboard refreshes up to ~7 times/second. Camera capture runs in a separate thread; slow detection skips old frames.")

            @st.fragment(run_every=0.15)
            def render_live_feed():
                latest = camera.get_latest()
                if latest.frame is None:
                    st.info(f"Camera: {latest.status}. {latest.error or 'Waiting for the first frame...'}")
                    return
                if latest.age_seconds is not None and latest.age_seconds > 3:
                    st.warning(f"Last frame was {latest.age_seconds:.1f} s ago. {latest.status}. {latest.error or ''}")
                height, width = latest.frame.shape[:2]
                frame_now = prepare_frame(latest.frame, cfg.processing_width)
                cache_key = (id(camera), latest.sequence, cfg)
                cached = st.session_state.get("last_live_result")
                if cached is not None and cached["key"] == cache_key:
                    annotated_now, debug_now, count, detection_ms = (
                        cached["annotated"], cached["debug"], cached["count"], cached["elapsed_ms"]
                    )
                else:
                    started = time.perf_counter()
                    hits, debug_now = detect(frame_now, cfg)
                    detection_ms = (time.perf_counter() - started) * 1000
                    annotated_now = draw_detections(frame_now, hits, cfg)
                    count = len(hits)
                    st.session_state["last_live_result"] = {
                        "key": cache_key, "annotated": annotated_now, "debug": debug_now,
                        "count": count, "elapsed_ms": detection_ms,
                    }
                a, b, c, d = st.columns(4)
                a.metric("Candidates", count)
                b.metric("Result", "Detected" if count else "Not detected")
                c.metric("Detection", f"{detection_ms:.0f} ms")
                d.metric("Camera decode rate", f"{latest.capture_fps:.1f} fps")
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Camera frame**")
                    st.image(frame_now, channels="BGR", use_container_width=True)
                with cols[1]:
                    st.markdown(f"**Annotated · {MODEL_NAME}**")
                    st.image(annotated_now, channels="BGR", use_container_width=True)
                st.caption(
                    f"Frame #{latest.sequence} · {width}×{height} source → "
                    f"{frame_now.shape[1]}×{frame_now.shape[0]} detection · "
                    f"{latest.status} · newest frame age: {latest.age_seconds or 0:.2f}s. "
                    "Scores are heuristic shape matches, not calibrated classification probabilities."
                )
                if show_debug:
                    debug_cols = st.columns(2)
                    debug_cols[0].image(debug_now["threshold_mask"], caption="RGB threshold mask", use_container_width=True)
                    debug_cols[1].image(debug_now["white_mask"], caption="Cleaned mask used by white detector", use_container_width=True)

            render_live_feed()
        st.stop()

    if source in ("Dataset images", "Sample image"):

        directory = IMAGE_DIR if source == "Dataset images" else SAMPLE_DIR
        options = list_media(directory, IMAGE_EXTENSIONS)
        if options:
            chosen = st.selectbox(
                f"Select image from {directory.relative_to(ROOT)}/",
                options,
                format_func=lambda p: str(p.relative_to(directory)),
            )
            frame = cv2.imread(str(chosen))
            frame_description = str(chosen.relative_to(ROOT))
            if frame is None:
                st.error("OpenCV could not decode this image. Check the file format and permissions.")
        else:
            st.info(
                f"No compatible images in `{directory.relative_to(ROOT)}/`. "
                "Add JPG, PNG, BMP, WebP, or TIFF images there. Subfolders are supported."
            )

    elif source in ("Dataset videos", "Sample video", "Upload video"):
        video_path = None
        if source == "Dataset videos":
            options = list_media(VIDEO_DIR, VIDEO_EXTENSIONS)
            if options:
                video_path = st.selectbox(
                    "Select video from Dataset/Videos/", options,
                    format_func=lambda p: str(p.relative_to(VIDEO_DIR)),
                )
            else:
                st.info("No videos in `Dataset/Videos/`. Add an MP4, MOV, AVI, MKV, WebM or M4V file.")
        elif source == "Sample video":
            video_path = SAMPLE_DIR / "05_moving_disc_demo.mp4"
            if not video_path.is_file():
                st.warning("Sample video missing. Run `python -m sample_generator.make_samples`. ")
                video_path = None
        else:
            upload = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "m4v", "mkv", "webm"])
            if upload is not None:
                video_path = saved_upload_path(upload.getvalue(), upload.name)

        if video_path is not None:
            info = video_info(video_path)
            if info is None:
                st.error("Cannot seek this video with OpenCV. Check the video codec/container.")
            else:
                identity = f"{video_path.resolve()}:{video_path.stat().st_mtime_ns}:{video_path.stat().st_size}"
                index = select_video_frame(identity, info.frame_count, info.fps)
                frame = read_video_frame(video_path, index)
                if frame is None:
                    st.error(f"Could not decode frame {index}. Try a different frame or an H.264 MP4.")
                frame_description = f"{video_path.name} · frame {index + 1}/{info.frame_count}"
                st.caption(f"Video: {info.width}×{info.height} · {info.fps:.2f} FPS metadata. Detection runs on the selected frame only.")

    elif source == "Upload image":
        upload = st.file_uploader("Upload a JPG or PNG", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if upload is not None:
            frame = image_from_bytes(upload.getvalue())
            frame_description = upload.name
            if frame is None:
                st.error("OpenCV could not decode the uploaded image.")

    elif source == "Camera snapshot":
        photo = st.camera_input("Take a photo")
        if photo is not None:
            frame = image_from_bytes(photo.getvalue())
            frame_description = "Camera snapshot"

    if frame is None:
        st.stop()

    frame = prepare_frame(frame, cfg.processing_width)
    start = time.perf_counter()
    candidates, debug = detect(frame, cfg)
    elapsed_ms = (time.perf_counter() - start) * 1000
    annotated = draw_detections(frame, candidates, cfg)

    count_col, state_col, latency_col = st.columns([1, 2, 1])
    count_col.metric("Candidates", len(candidates))
    state_col.metric("Result", "Candidate detected" if candidates else "No candidate")
    latency_col.metric("Detection time", f"{elapsed_ms:.0f} ms")

    col_input, col_output = st.columns(2)
    with col_input:
        st.subheader("Input")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_output:
        st.subheader(f"Results · {MODEL_NAME}")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.caption(f"{frame_description} · detection frame {frame.shape[1]}×{frame.shape[0]}")
    st.subheader("Candidate details")
    if candidates:
        st.dataframe([
            {
                "#": i,
                "Found by": c.method,
                "Center X": c.center[0], "Center Y": c.center[1],
                "Axis X (px)": c.axes[0], "Axis Y (px)": c.axes[1],
                "Match score (heuristic)": round(c.score, 3),
            }
            for i, c in enumerate(candidates, start=1)
        ], hide_index=True, use_container_width=True)
    else:
        st.info("No candidate meets the current settings. Adjust the white thresholds or size limits.")

    with st.expander("Debug views: RGB threshold and cleaned mask", expanded=True):
        st.caption("Pixels pass when R + G + B ≥ the minimum AND max(R,G,B) − min(R,G,B) ≤ the maximum difference. The ellipse detector uses the cleaned mask.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**RGB threshold mask**")
            st.image(debug["threshold_mask"], clamp=True, use_container_width=True)
        with c2:
            st.markdown("**Cleaned detection mask**")
            st.image(debug["white_mask"], clamp=True, use_container_width=True)

    ok, png = cv2.imencode(".png", annotated)
    if ok:
        st.download_button("Download annotated image", png.tobytes(), "ground_frisbee_result.png", "image/png")

    st.caption("Limit: white ellipses such as plates, markings and lids may trigger; the detector cannot verify physical contact with the ground.")
