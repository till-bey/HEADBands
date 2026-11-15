import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detector import DetectorSettings, detect, draw_detections, preprocess
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CFG = DetectorSettings()


def sample(name: str):
    frame = cv2.imread(str(ROOT / "samples" / name))
    assert frame is not None
    return frame


def test_ellipse_finds_angled_disc():
    results, _ = detect(sample("02_angled_white_disc.png"), CFG)
    assert len(results) >= 1
    assert abs(results[0].center[0] - 415) < 25
    assert abs(results[0].center[1] - 330) < 25


def test_ellipse_finds_round_disc():
    results, _ = detect(sample("01_round_white_disc.png"), CFG)
    assert len(results) >= 1
    assert abs(results[0].center[0] - 390) < 25
    assert abs(results[0].center[1] - 330) < 25


def test_blank_field_has_no_detection():
    results, _ = detect(sample("03_empty_field.png"), CFG)
    assert not results


def test_ellipse_draws_results():
    frame = sample("04_two_white_discs.png")
    results, debug = detect(frame, CFG)
    assert len(results) >= 2
    assert set(debug) == {"threshold_mask", "white_mask"}
    annotated = draw_detections(frame, results, CFG)
    assert annotated.shape == frame.shape


def test_two_discs_have_exactly_two_detections():
    results, _ = detect(sample("04_two_white_discs.png"), CFG)
    assert len(results) == 2


def colored_disc(color):
    frame = np.full((240, 320, 3), (40, 100, 40), dtype=np.uint8)
    cv2.ellipse(frame, (160, 160), (50, 25), 20, 0, 360, color, -1)
    return frame


def test_bright_colors_are_not_white():
    # Pale red used to pass HSV V>=165, S<=90, alongside saturated distractors.
    for color in [(180, 180, 255), (0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]:
        frame = colored_disc(color)
        results, debug = detect(frame, CFG)
        assert debug["white_mask"][160, 160] == 0, color
        assert not results, color


def test_neutral_white_shadow_and_mild_warm_cast_are_detected():
    for color in [(245, 245, 245), (145, 145, 145), (180, 190, 200)]:
        results, debug = detect(colored_disc(color), CFG)
        assert debug["white_mask"][160, 160] == 255, color
        assert len(results) == 1, color


def test_lower_rgb_sum_includes_shadows_without_accepting_red():
    assert not detect(colored_disc((110, 110, 110)), CFG)[0]
    cfg = DetectorSettings(rgb_sum_min=300)
    assert len(detect(colored_disc((110, 110, 110)), cfg)[0]) == 1
    assert not detect(colored_disc((100, 100, 210)), cfg)[0]


def test_channel_difference_control_and_ground_roi():
    frame = colored_disc((180, 180, 255))
    strict, _ = preprocess(frame, CFG)
    relaxed, _ = preprocess(frame, DetectorSettings(channel_difference_max=80))
    assert strict[160, 160] == 0
    assert relaxed[160, 160] == 255
    white = colored_disc((240, 240, 240))
    raw, mask = preprocess(white, DetectorSettings(ground_start_pct=90))
    assert not mask[:216].any()
    assert not raw[:216].any()


def test_detection_uses_cleaned_threshold_mask():
    frame = colored_disc((240, 240, 240))
    raw, mask = preprocess(frame, CFG)
    assert detect(frame, CFG)[0]
    with patch("detector.preprocess", return_value=(raw, np.zeros_like(mask))):
        assert not detect(frame, CFG)[0]


def test_rgb_threshold_boundaries_and_no_uint8_overflow():
    # Exact inclusive boundaries, all channel permutations, and bright white.
    for color, minimum, maximum, passes in [
        ((140, 145, 150), 435, 10, True),
        ((150, 140, 145), 435, 10, True),
        ((140, 145, 150), 436, 10, False),
        ((140, 145, 150), 435, 9, False),
        ((255, 255, 255), 765, 0, True),
        ((0, 0, 255), 0, 254, False),
        ((0, 0, 255), 0, 255, True),
    ]:
        frame = np.full((12, 12, 3), color, dtype=np.uint8)
        cfg = DetectorSettings(ground_start_pct=0, rgb_sum_min=minimum, channel_difference_max=maximum)
        raw, cleaned = preprocess(frame, cfg)
        assert bool(raw[6, 6]) == passes, (color, minimum, maximum)
        assert bool(cleaned[6, 6]) == passes


def test_prepare_frame_landscape_portrait_and_small_sources():
    from detector import prepare_frame
    for source_shape, target_shape in [
        ((1200, 1600, 3), (480, 640, 3)),
        ((1600, 1200, 3), (640, 480, 3)),
        ((240, 320, 3), (480, 640, 3)),
        ((320, 240, 3), (640, 480, 3)),
    ]:
        frame = np.full(source_shape, 200, dtype=np.uint8)
        result = prepare_frame(frame)
        assert result.shape == target_shape
        assert np.all(result == 200)
        assert frame.shape == source_shape


def test_prepare_frame_preserves_geometry_with_letterboxing():
    from detector import prepare_frame
    for height, width in [(900, 1600), (1600, 900), (500, 500)]:
        frame = np.full((height, width, 3), 100, dtype=np.uint8)
        cv2.circle(frame, (width // 2, height // 2), 100, (255, 255, 255), -1)
        result = prepare_frame(frame)
        ys, xs = np.where(result[:, :, 0] > 200)
        assert abs((xs.max() - xs.min()) - (ys.max() - ys.min())) <= 1
        assert not result[0, 0].any()  # padding instead of distortion/cropping
        assert prepare_frame(result) is result  # prepared frames are unchanged


def test_prepared_frame_detection_and_annotation_share_dimensions():
    from detector import prepare_frame
    frame = prepare_frame(sample('02_angled_white_disc.png'))
    candidates, debug = detect(frame, CFG)
    assert candidates
    assert frame.shape == (480, 640, 3)
    assert debug['white_mask'].shape == debug['threshold_mask'].shape == frame.shape[:2]
    assert draw_detections(frame, candidates, CFG).shape == frame.shape
