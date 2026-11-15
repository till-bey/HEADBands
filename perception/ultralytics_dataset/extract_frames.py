import os
import subprocess
import json
from pathlib import Path
import shutil

def get_video_duration(video_path):
    """Get the video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    data = json.loads(result.stdout)
    return float(data['format']['duration'])

def extract_screenshots(input_dir, output_dir, num_screenshots=10):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported video extensions
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
    videos = [p for p in input_path.iterdir() if p.suffix.lower() in video_extensions]

    if not videos:
        print(f"No video files found in '{input_dir}'.")
        return

    for video in videos:
        print(f"Processing: {video.name}")
        try:
            duration = get_video_duration(video)
        except Exception as e:
            print(f"Skipping {video.name} — could not read duration: {e}")
            continue

        # Divide the video duration into equal intervals
        interval = duration / (num_screenshots + 1)
        
        # Create a dedicated output subfolder for each video
        vid_out_dir = output_path / video.stem
        vid_out_dir.mkdir(exist_ok=True)

        for i in range(1, num_screenshots + 1):
            timestamp = interval * i
            out_file = vid_out_dir / f"shot_{i:02d}.jpg"
            
            # FFmpeg command (-ss before -i for fast seeking)
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", str(video),
                "-vframes", "1",
                "-q:v", "2",  # High JPEG quality (2 is high, lower is better)
                str(out_file)
            ]
            
            # Run quietly without flooding the console
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"Saved {num_screenshots} screenshots to: {vid_out_dir}")

    print("\nAll extractions complete!")

def flatten_screenshots(source_dir, flat_output_dir):
    source_path = Path(source_dir)
    output_path = Path(flat_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all subdirectories
    subfolders = [p for p in source_path.iterdir() if p.is_dir()]

    if not subfolders:
        print(f"No subfolders found in '{source_dir}'.")
        return

    count = 0
    for folder in subfolders:
        video_name = folder.name
        # Find all images inside the subfolder
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

        for img in images:
            # Create a unique, simple name: videoname_shot_01.jpg
            new_filename = f"{video_name}_{img.name}"
            dest_file = output_path / new_filename
            
            # Copy the file to the flat folder
            shutil.copy2(img, dest_file)
            count += 1

    print(f"Successfully flattened {count} screenshots into: {output_path}")

if __name__ == "__main__":
    # Configure your folders here
    INPUT_FOLDER = Path(__file__).parent / "videos"
    OUTPUT_FOLDER = Path(__file__).parent / "screenshots"
    
    # extract_screenshots(INPUT_FOLDER, OUTPUT_FOLDER, num_screenshots=10)
    flatten_screenshots(OUTPUT_FOLDER, OUTPUT_FOLDER)