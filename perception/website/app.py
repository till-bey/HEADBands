"""Streamlit UI for ground frisbee detection on folders, video and live IP streams."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Streamlit adds website/ to the import path, but sibling packages live in
# perception/. Resolve from this file so launching from any directory works.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector import DetectorSettings, detect, draw_detections, prepare_frame, RGB_MEAN_RADIUS_SCALE
from website.ground_position import render_ground_position
from website.saved_settings import DEFAULT_SETTINGS_FILE, DEFAULT_DASHBOARD_URL, load_navigation_settings
from camera import LatestFrameCamera, STREAM_MODES, VIDEO_STREAM, SNAPSHOT
from website.media_utils import (
    IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, list_media, read_video_frame, video_info,
)

# Media paths use the same absolute root as the imports.
IMAGE_DIR = ROOT / "Dataset" / "Images"
VIDEO_DIR = ROOT / "Dataset" / "Videos"
SAMPLE_DIR = ROOT / "samples"
MODEL_NAME = "Frisbee rims and discs"

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


def show_candidate_details(candidates) -> None:
    st.subheader("Candidate details")
    if not candidates:
        st.info("No candidate meets the current settings. Adjust rim edge thresholds or size limits.")
        return
    st.dataframe([
        {"#": i, "Found by": candidate.method,
         "Center X": candidate.center[0], "Center Y": candidate.center[1],
         "Axis X (px)": candidate.axes[0], "Axis Y (px)": candidate.axes[1],
         "Match score (heuristic)": round(candidate.score, 3),
         "Mean R+G+B (0–765)": round(candidate.mean_rgb_sum, 2)}
        for i, candidate in enumerate(candidates, start=1)
    ], hide_index=True, use_container_width=True)
    st.caption("Mean R+G+B uses the inner 75% of both ellipse radii, excluding the outer rim. All pixels in this inner ellipse are included, including dark markings. Range: 0–765.")


with st.sidebar:
    st.header("Shared saved configuration")
    with st.expander("Configuration source"):
        shared_path = st.text_input("Navigation settings file", str(DEFAULT_SETTINGS_FILE), key="shared_config_path")
        dashboard_url = st.text_input("Navigation dashboard URL", DEFAULT_DASHBOARD_URL, key="shared_dashboard_url")
    reload_shared = st.button("Reload saved configuration", use_container_width=True)
    source_identity = (shared_path, dashboard_url)
    if ("shared_navigation_values" not in st.session_state or reload_shared
            or st.session_state.get("shared_source_identity") != source_identity):
        try:
            loaded, loaded_source = load_navigation_settings(shared_path, dashboard_url)
            st.session_state["shared_navigation_values"] = loaded
            st.session_state["shared_navigation_source"] = loaded_source
            st.session_state.pop("shared_navigation_error", None)
            # Reset only calibration/detector form widgets. Camera readers remain
            # open until the user explicitly reconnects with the newly loaded URL.
            for key in list(st.session_state):
                if key.startswith(("det_", "ground_cfg_", "live_cfg_")):
                    del st.session_state[key]
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            st.session_state["shared_navigation_error"] = str(exc)
            st.session_state.setdefault("shared_navigation_values", {})
        st.session_state["shared_source_identity"] = source_identity
    saved_configuration = st.session_state["shared_navigation_values"]
    if "shared_navigation_error" in st.session_state:
        st.warning(st.session_state["shared_navigation_error"])
    else:
        st.caption("Loaded: " + st.session_state["shared_navigation_source"])
    st.caption("Read-only: Streamlit adjustments are temporary. Save in the navigation dashboard, then reload here. Reconnect cameras after changing their saved URL or format.")
    defaults = DetectorSettings()
    st.header("Detection settings")
    processing_width = st.selectbox(
        "Processing resolution", [320, 480, 640, 960, 1280],
        index=[320, 480, 640, 960, 1280].index(saved_configuration.get("resolution", 640)), key="det_resolution",
        format_func=lambda w: f"{w}×{w * 3 // 4}",
        help="Portrait swaps the dimensions. Lower sizes reduce work but can lose small discs. Radius and area settings use these pixels.",
    )
    st.caption("For speed, start with 320×240.")
    st.subheader("Rim detection")
    st.caption("Fit ellipses to the frisbee's outer edges, including tilted side views.")
    canny_low = st.slider("Rim edge lower threshold", 0, 254, saved_configuration.get("canny_low", defaults.canny_low), key="det_canny_low")
    canny_high = st.slider("Rim edge upper threshold", 1, 255, saved_configuration.get("canny_high", defaults.canny_high), key="det_canny_high")
    st.subheader("White-region fallback")
    rgb_sum_min = st.slider(
        "Minimum R + G + B", 0, 765, saved_configuration.get("rgb_sum_min", defaults.rgb_sum_min), 1, key="det_rgb_sum_min",
        help="Require R + G + B to reach this value. Lower to include white discs in shadow.",
    )
    channel_difference_max = st.slider(
        "Maximum channel difference", 0, 255, saved_configuration.get("channel_difference_max", defaults.channel_difference_max), 1, key="det_channel_difference_max",
        help="Require max(R, G, B) − min(R, G, B) to stay below this value. This limits every pairwise channel difference; lower to reject tinted objects.",
    )
    min_white_fraction = st.number_input("Minimum white fill fraction", min_value=0.000001, max_value=1.0, value=saved_configuration.get("min_white_fraction", defaults.min_white_fraction), step=0.01, format="%.4f", key="det_min_white_fraction")
    with st.expander("Ellipse size and shape tuning"):
        min_radius = st.number_input("Minimum approximate radius (px)", min_value=1, value=saved_configuration.get("min_radius", defaults.min_radius), step=1, key="det_min_radius")
        max_radius = st.number_input("Maximum approximate radius (px)", min_value=2, value=saved_configuration.get("max_radius", defaults.max_radius), step=1, key="det_max_radius")
        min_ellipse_ratio = st.number_input("Minimum ellipse aspect ratio", min_value=0.000001, max_value=1.0, value=saved_configuration.get("min_ellipse_ratio", defaults.min_ellipse_ratio), step=0.01, format="%.4f", key="det_min_ellipse_ratio")
        min_area = st.number_input("Minimum contour area (px²)", min_value=1, value=saved_configuration.get("min_area", defaults.min_area), step=1, key="det_min_area")

if canny_low >= canny_high or min_radius >= max_radius:
    st.error("Lower rim threshold and minimum radius must be below their corresponding upper values.")
    st.stop()

cfg = DetectorSettings(
    canny_low=canny_low, canny_high=canny_high,
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
st.write("Find frisbee rims and elliptical discs with classical computer vision, then measure their average R+G+B.")
st.caption(f"Processing at {processing_width}×{processing_width * 3 // 4} landscape, with dimensions swapped for portrait and padding to preserve proportions. Pixel radius/area limits apply at this size.")

detection_tab, position_tab = st.tabs(["Detection", "Ground position"])
# Render the independent dashboard before the existing detector's early stops.
with position_tab:
    render_ground_position(cfg, saved_configuration=saved_configuration)

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
                value=saved_configuration.get("url", ""), key="live_cfg_url",
                placeholder="http://192.168.1.50:8080/?action=stream",
                help="Examples: http://IP:8080/?action=stream | http://IP:8080/stream.mjpg | rtsp://IP:8554/cam",
            )
            camera_mode = st.selectbox("Stream format", STREAM_MODES, index=int(saved_configuration.get("snapshot", False)), key="live_cfg_mode")
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
                cache_key = (id(camera), latest.sequence, cfg, RGB_MEAN_RADIUS_SCALE)
                cached = st.session_state.get("last_live_result")
                if cached is not None and cached["key"] == cache_key and "candidates" in cached:
                    candidates_now = cached["candidates"]
                    annotated_now, debug_now, count, detection_ms = (
                        cached["annotated"], cached["debug"], cached["count"], cached["elapsed_ms"]
                    )
                else:
                    started = time.perf_counter()
                    hits, debug_now = detect(frame_now, cfg)
                    candidates_now = hits
                    detection_ms = (time.perf_counter() - started) * 1000
                    annotated_now = draw_detections(frame_now, hits, cfg)
                    count = len(hits)
                    st.session_state["last_live_result"] = {
                        "key": cache_key, "annotated": annotated_now, "debug": debug_now,
                        "count": count, "elapsed_ms": detection_ms,
                        "candidates": candidates_now,
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
                show_candidate_details(candidates_now)
                if show_debug:
                    debug_cols = st.columns(3)
                    debug_cols[0].image(debug_now["threshold_mask"], caption="RGB threshold mask", use_container_width=True)
                    debug_cols[1].image(debug_now["white_mask"], caption="Cleaned mask used by white detector", use_container_width=True)
                    debug_cols[2].image(debug_now["edge_mask"], caption="Rim edges used for ellipse fitting", use_container_width=True)

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
    show_candidate_details(candidates)

    with st.expander("Debug views: rim edges and white masks", expanded=True):
        st.caption("Rim ellipses use image edges independently of interior color. The white-region fallback uses the RGB threshold and cleaned mask.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**RGB threshold mask**")
            st.image(debug["threshold_mask"], clamp=True, use_container_width=True)
        with c2:
            st.markdown("**Cleaned detection mask**")
            st.image(debug["white_mask"], clamp=True, use_container_width=True)
        with c3:
            st.markdown("**Rim edges**")
            st.image(debug["edge_mask"], clamp=True, use_container_width=True)

    ok, png = cv2.imencode(".png", annotated)
    if ok:
        st.download_button("Download annotated image", png.tobytes(), "ground_frisbee_result.png", "image/png")

    st.caption("Limit: other elliptical rims, plates, markings and lids may trigger; shape alone cannot identify a frisbee or verify contact with the ground.")
