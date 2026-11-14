"""Download, fine-tune and run YOLO frisbee detection/tracking.

Dataset: https://universe.roboflow.com/dustin-klein/dg-putting/dataset/1
Classes of interest: disc_flying and disc_ground.

Examples:
    python frisbee_pipeline.py download
    python frisbee_pipeline.py train --epochs 50 --imgsz 960 --batch 4
    python frisbee_pipeline.py evaluate
    python frisbee_pipeline.py image --source data/dg-putting-v1/test/images/example.jpg
    python frisbee_pipeline.py video --source 0
    python frisbee_pipeline.py video --source match.mp4 --save results/tracked.mp4 --no-show
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import cv2
import torch
import yaml
from ultralytics import YOLO

WORKSPACE = "dustin-klein"
PROJECT = "dg-putting"
VERSION = 1
DATA_DIR = Path("data/dg-putting-v1")
DEFAULT_MODEL = "yolo26s.pt"
DEFAULT_RUN_NAME = "yolo26s_960"
DEFAULT_WEIGHTS = Path("runs/frisbee") / DEFAULT_RUN_NAME / "weights/best.pt"
TARGET_CLASSES = ("disc_flying", "disc_ground")
COLORS = {"disc_flying": (255, 150, 40), "disc_ground": (55, 210, 70)}  # BGR


def choose_device() -> str:
    """Use CUDA GPU if present; fall back to CPU."""
    return "0" if torch.cuda.is_available() else "cpu"


def _find_dataset_root(location: Path) -> Path:
    """Find the actual Roboflow export folder, including nested exports."""
    for candidate in (location, *sorted(location.glob("*/"))):
        if (candidate / "data.yaml").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find data.yaml under {location}")


def prepare_data_yaml(dataset_dir: str | Path) -> Path:
    """Create a local, absolute-path YOLO YAML, avoiding Roboflow path quirks."""
    root = _find_dataset_root(Path(dataset_dir).expanduser())
    with (root / "data.yaml").open("r", encoding="utf-8") as f:
        original = yaml.safe_load(f)

    split_folders = {"train": "train", "val": "valid", "test": "test"}
    if not (root / "valid/images").is_dir() and (root / "val/images").is_dir():
        split_folders["val"] = "val"

    required = ("train", "val")
    for key in required:
        if not (root / split_folders[key] / "images").is_dir():
            raise FileNotFoundError(f"Expected {key} images under {root}")

    config: dict[str, Any] = {
        "path": str(root),
        "train": f"{split_folders['train']}/images",
        "val": f"{split_folders['val']}/images",
        "names": original["names"],
    }
    if (root / "test/images").is_dir():
        config["test"] = "test/images"
    config["nc"] = len(original["names"])

    yaml_path = root / "data_local.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    names = original["names"]
    name_values = names.values() if isinstance(names, dict) else names
    missing = sorted(set(TARGET_CLASSES) - {str(n).lower() for n in name_values})
    if missing:
        raise ValueError(f"Expected frisbee classes missing from dataset: {missing}")
    return yaml_path


def download_dataset(
    output_dir: str | Path = DATA_DIR,
    api_key: str | None = None,
) -> Path:
    """Download Roboflow DG Putting v1 (requires an API key), return local YAML.

    Reads the API key from ROBOFLOW_API_KEY if `api_key` is omitted.
    Skips the download when the dataset already exists.
    """
    output_dir = Path(output_dir).expanduser()
    if (output_dir / "data.yaml").is_file():
        print(f"Dataset exists; reusing {output_dir}")
        return prepare_data_yaml(output_dir)

    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing Roboflow API key. Set ROBOFLOW_API_KEY or pass api_key=. "
            "Get a key from your Roboflow account settings."
        )

    from roboflow import Roboflow

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(WORKSPACE)
        .project(PROJECT)
        .version(VERSION)
        .download("yolov8", location=str(output_dir.resolve()))
    )
    print(f"Downloaded to {dataset.location}")
    return prepare_data_yaml(dataset.location)


def train_model(
    data_yaml: str | Path,
    model_name: str = DEFAULT_MODEL,
    epochs: int = 50,
    imgsz: int = 960,
    batch: int = 4,
    run_name: str = DEFAULT_RUN_NAME,
    device: str | None = None,
) -> Path:
    """Fine-tune pretrained weights and return the best checkpoint path."""
    data_yaml = Path(data_yaml).expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=15,
        project="runs/frisbee",
        name=run_name,
        exist_ok=True,
        device=device or choose_device(),
        workers=0,  # works in notebooks and on Windows too
        seed=42,
    )
    best = Path(model.trainer.best).resolve()
    if not best.is_file():
        raise FileNotFoundError(f"Training did not create expected checkpoint: {best}")
    print(f"Best trained weights: {best}")
    return best


def evaluate_model(
    weights: str | Path,
    data_yaml: str | Path,
    imgsz: int = 960,
    split: str = "test",
    device: str | None = None,
) -> dict[str, float]:
    """Evaluate on train/val/test, defaulting to held-out test split."""
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(Path(data_yaml).resolve()),
        imgsz=imgsz,
        split=split,
        device=device or choose_device(),
        plots=True,
    )
    summary = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }
    print(f"{split} metrics: {summary}")
    return summary


def disc_class_ids(names: dict[int, str] | list[str]) -> list[int]:
    """Return class indices without assuming Roboflow's class ordering."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    ids = [int(idx) for idx, name in items if str(name).strip().lower() in TARGET_CLASSES]
    if not ids:
        raise ValueError(f"No {TARGET_CLASSES} classes found in model.names: {names}")
    return ids


def _draw_disc_boxes(frame, result) -> list[dict[str, Any]]:
    """Draw ONLY frisbee predictions; return simple serializable detections."""
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls.item())
        name = str(result.names[class_id])
        if name.lower() not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = (int(x) for x in box.xyxy[0].tolist())
        confidence = float(box.conf.item())
        track_id = int(box.id.item()) if box.id is not None else None
        color = COLORS[name.lower()]
        caption = f"{name} {confidence:.2f}"
        if track_id is not None:
            caption += f" id={track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, caption, (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )
        detections.append(
            {"class": name, "confidence": confidence,
             "xyxy": [x1, y1, x2, y2], "track_id": track_id}
        )
    return detections


def predict_image(
    weights: str | Path,
    image_path: str | Path,
    save_path: str | Path = "results/image_prediction.jpg",
    imgsz: int = 960,
    conf: float = 0.15,
) -> tuple[Path, list[dict[str, Any]]]:
    """Run single-image prediction and save an annotated image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    model = YOLO(str(weights))
    result = model.predict(
        image, imgsz=imgsz, conf=conf, classes=disc_class_ids(model.names),
        device=choose_device(), verbose=False,
    )[0]
    detections = _draw_disc_boxes(image, result)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(save_path), image):
        raise RuntimeError(f"Could not write {save_path}")
    print(f"Saved image to {save_path}. Detections: {detections}")
    return save_path, detections


def run_video(
    weights: str | Path,
    source: str | int = 0,
    imgsz: int = 960,
    conf: float = 0.10,
    show: bool = True,
    save_path: str | Path | None = None,
    tracker: str = "bytetrack.yaml",
    max_frames: int | None = None,
) -> None:
    """Run persistent tracking on webcam (0), video file, or stream.

    Press Q or ESC to close. For headless servers set show=False and save_path.
    """
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    model = YOLO(str(weights))
    ids = disc_class_ids(model.names)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 240 else 30.0
    last_time = time.perf_counter()
    frame_count = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            result = model.track(
                frame, persist=True, tracker=tracker,
                imgsz=imgsz, conf=conf, classes=ids,
                device=choose_device(), verbose=False,
            )[0]
            detections = _draw_disc_boxes(frame, result)
            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(1e-6, now - last_time)
            last_time = now
            cv2.putText(
                frame, f"Processing FPS: {instantaneous_fps:.1f} | Discs: {len(detections)}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA,
            )

            if save_path is not None:
                if writer is None:
                    save_file = Path(save_path)
                    save_file.parent.mkdir(parents=True, exist_ok=True)
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(save_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open video writer: {save_file}")
                writer.write(frame)

            if show:
                cv2.imshow("Frisbee detection — Q or ESC to quit", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames" +
          (f"; saved to {save_path}" if save_path else ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Download DG Putting v1")
    p.add_argument("--output", default=str(DATA_DIR))

    p = sub.add_parser("train", help="Fine-tune YOLO26s")
    p.add_argument("--data", default=str(DATA_DIR / "data_local.yaml"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--name", default=DEFAULT_RUN_NAME)

    p = sub.add_parser("evaluate", help="Evaluate checkpoint")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--data", default=str(DATA_DIR / "data_local.yaml"))
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--imgsz", type=int, default=960)

    p = sub.add_parser("image", help="Predict on one image")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--source", required=True)
    p.add_argument("--save", default="results/image_prediction.jpg")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.15)

    p = sub.add_parser("video", help="Track frisbees in webcam or video")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--source", default="0", help="Webcam index (0), file or stream URL")
    p.add_argument("--save", default=None, help="Optional output .mp4 path")
    p.add_argument("--no-show", action="store_true", help="Don't open OpenCV window")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--max-frames", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download":
        print(f"Local YOLO config: {download_dataset(args.output)}")
    elif args.command == "train":
        train_model(args.data, args.model, args.epochs, args.imgsz, args.batch, args.name)
    elif args.command == "evaluate":
        evaluate_model(args.weights, args.data, args.imgsz, args.split)
    elif args.command == "image":
        predict_image(args.weights, args.source, args.save, args.imgsz, args.conf)
    elif args.command == "video":
        run_video(args.weights, args.source, args.imgsz, args.conf,
                  not args.no_show, args.save, args.tracker, args.max_frames)


if __name__ == "__main__":
    main()
