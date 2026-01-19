#!/usr/bin/env python3
"""
Generate a per-frame blank mask sequence for a given video.

- Masks match the video's frame count and resolution.
- Files are saved as zero-padded PNGs: 0000.png, 0001.png, ...
 - Default padding is 4 digits; adjustable via --digits.
 - Default mask value is 0 (black); adjustable via --value 0..255.

Usage:
  python generate_blank_masks.py --video ./assets/input/woman.mp4 --out ./assets/mask/woman_blank

Optional:
  python generate_blank_masks.py --video v.mp4 --out out_dir --value 255 --digits 5
"""

import argparse
import os
import sys
import cv2
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_blank_masks(video_path: str, out_dir: str, value: int = 0, digits: int = 4) -> int:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if digits < 1:
        raise ValueError("digits must be >= 1")
    if not (0 <= value <= 255):
        raise ValueError("value must be an integer in [0, 255]")

    ensure_dir(out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    frame_idx = 0
    height = width = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if height is None:
                height, width = frame.shape[:2]
                if height is None or width is None:
                    raise RuntimeError("Failed to read frame size from video.")

            # Create a blank grayscale mask matching frame size
            mask = np.full((height, width), int(value), dtype=np.uint8)

            out_path = os.path.join(out_dir, f"{frame_idx:0{digits}d}.png")
            ok = cv2.imwrite(out_path, mask)
            if not ok:
                raise RuntimeError(f"Failed to write mask: {out_path}")

            frame_idx += 1
    finally:
        cap.release()

    if frame_idx == 0:
        raise RuntimeError("No frames decoded from the input video.")

    return frame_idx


def parse_args():
    p = argparse.ArgumentParser(description="Generate blank mask PNG sequence for a video")
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--out", required=True, help="Output folder for the PNG masks")
    p.add_argument("--value", type=int, default=0, help="Mask pixel value (0..255), default 0 (black)")
    p.add_argument("--digits", type=int, default=4, help="Zero-padding width for filenames, default 4")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        count = generate_blank_masks(args.video, args.out, args.value, args.digits)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Wrote {count} masks to '{args.out}'.")


if __name__ == "__main__":
    main()
