#!/usr/bin/env python3
"""
Batch video processor (OpenCV only, no audio):
- Change FPS (resample by duplicating/dropping frames to preserve duration).
- Trim to a target length or split into fixed-length segments (last segment kept even if short).
- Support multiple input videos; outputs are written to a target folder.

Example usages:
  # Trim to 6s and set FPS to 12 for multiple videos
  python batch_video_process.py --videos ./a.mp4 ./b.mp4 --out-dir ./output --seconds 6 --fps 12 --backend ffmpeg --fourcc mp4v

  # Split each video into 2.5s chunks, keep original FPS, try safer fallbacks
  python batch_video_process.py --videos ./assets/LIGHT/talk/talk2srfps.mp4 \
    --out-dir ./output/splits --split-seconds 2.5 --fallback-fourccs mp4v,XVID,MJPG --backend ffmpeg

Notes:
- Audio is NOT preserved (OpenCV writer only).
- If H.264 is unavailable, use mp4v/XVID/MJPG or re-encode input with ffmpeg.
"""

import argparse
import math
import os
import sys
from typing import Iterable, List, Sequence, Tuple

import cv2


def _resolve_backend(name: str) -> int:
    name = name.lower()
    mapping = {
        "any": cv2.CAP_ANY,
        "ffmpeg": cv2.CAP_FFMPEG,
        "gstreamer": cv2.CAP_GSTREAMER,
        "avfoundation": cv2.CAP_AVFOUNDATION,
        "msmf": cv2.CAP_MSMF,
        "dshow": cv2.CAP_DSHOW,
    }
    return mapping.get(name, cv2.CAP_ANY)


def _open_video(path: str, backend: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path, _resolve_backend(backend))
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {path}. Try --backend ffmpeg or re-encode to H.264 mp4."
        )
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise RuntimeError(
            f"Failed to decode first frame from: {path}. "
            "Re-encode (ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4) or use --backend ffmpeg."
        )
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap


def _make_writer(path: str, fps: float, size: Tuple[int, int], fourcc: str, fallback_fourccs: Sequence[str]) -> cv2.VideoWriter:
    tried: List[str] = []
    for code in [fourcc] + [c for c in fallback_fourccs if c and c != fourcc]:
        tried.append(code)
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*code),
            fps,
            size,
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        f"Cannot open writer for output: {path}. Tried fourcc: {tried}. "
        "Install ffmpeg with H.264 (libx264) support or use mp4v/XVID/MJPG."
    )


def _segments(total_frames: int, fps: float, split_seconds: float | None, trim_seconds: float | None) -> List[Tuple[int, int]]:
    if total_frames <= 0:
        raise RuntimeError("No frames in video")

    if split_seconds is not None:
        if split_seconds <= 0:
            raise ValueError("--split-seconds must be positive")
        frames_per = max(1, int(round(split_seconds * fps)))
        count = math.ceil(total_frames / frames_per)
        segs = []
        for i in range(count):
            start = i * frames_per
            end = min(total_frames, (i + 1) * frames_per)
            segs.append((start, end))
        return segs

    if trim_seconds is not None:
        if trim_seconds <= 0:
            raise ValueError("--seconds must be positive")
        end = min(total_frames, int(round(trim_seconds * fps)))
        end = max(1, end)
        return [(0, end)]

    # default: full video
    return [(0, total_frames)]


def _process_one(
    video_path: str,
    out_dir: str,
    target_fps: float | None,
    trim_seconds: float | None,
    split_seconds: float | None,
    backend: str,
    fourcc: str,
    fallback_fourccs: Sequence[str],
):
    base = os.path.splitext(os.path.basename(video_path))[0]
    cap = _open_video(video_path, backend)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read video width/height")

    segs = _segments(total_frames, src_fps, split_seconds, trim_seconds)

    # Accumulator-based resampling works for both down/up FPS while preserving duration
    out_paths: List[str] = []
    for idx, (start, end) in enumerate(segs, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        out_path = os.path.join(out_dir, f"{base}_part{idx:03d}.mp4")
        write_fps = target_fps if target_fps and target_fps > 0 else src_fps
        factor = write_fps / src_fps
        writer = _make_writer(out_path, write_fps, (width, height), fourcc, fallback_fourccs)

        # Expected frames for this segment based on desired output FPS to reduce off-by-one drops
        seg_duration = (end - start) / src_fps if src_fps > 0 else 0
        expected_out = max(1, int(round(seg_duration * write_fps))) if seg_duration > 0 else 1

        acc = 0.0
        written = 0
        last_frame = None
        try:
            for frame_idx in range(start, end):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                acc += factor
                while acc >= 1.0:
                    writer.write(frame)
                    last_frame = frame
                    acc -= 1.0
                    written += 1
            # If rounding caused one-frame shortfall, pad with last frame
            if written < expected_out and last_frame is not None:
                while written < expected_out:
                    writer.write(last_frame)
                    written += 1
        finally:
            writer.release()

        out_paths.append(out_path)

    cap.release()
    return out_paths


def parse_args():
    p = argparse.ArgumentParser(description="Batch video FPS change, trim or split (OpenCV only, no audio)")
    p.add_argument("--videos", nargs="+", required=True, help="Input video paths (one or more)")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--fps", type=float, help="Target FPS (if omitted, keep source FPS)")
    p.add_argument("--seconds", type=float, help="Trim to this length in seconds (optional)")
    p.add_argument("--split-seconds", type=float, help="Split into fixed-length segments in seconds (optional)")
    p.add_argument("--fourcc", default="mp4v", help="FOURCC for VideoWriter (default mp4v; try avc1 for H.264)")
    p.add_argument(
        "--fallback-fourccs",
        default="mp4v,XVID,MJPG",
        help="Comma-separated fallback fourcc list tried if the first fails",
    )
    p.add_argument(
        "--backend",
        default="any",
        help="VideoCapture backend: any, ffmpeg, gstreamer, avfoundation, msmf, dshow (default any)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.seconds is not None and args.split_seconds is not None:
        print("Error: use either --seconds (trim) or --split-seconds (segment), not both", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    fallbacks = [c.strip() for c in args.fallback_fourccs.split(",") if c.strip()]

    all_outputs: List[str] = []
    for vid in args.videos:
        try:
            outputs = _process_one(
                video_path=vid,
                out_dir=out_dir,
                target_fps=args.fps,
                trim_seconds=args.seconds,
                split_seconds=args.split_seconds,
                backend=args.backend,
                fourcc=args.fourcc,
                fallback_fourccs=fallbacks,
            )
            all_outputs.extend(outputs)
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing {vid}: {exc}", file=sys.stderr)
            continue

    if not all_outputs:
        print("No outputs produced (all inputs failed)", file=sys.stderr)
        sys.exit(1)

    print("Done. Outputs:")
    for pth in all_outputs:
        print(pth)


if __name__ == "__main__":
    main()
