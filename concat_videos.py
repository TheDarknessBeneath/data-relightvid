#!/usr/bin/env python3
"""
Concatenate multiple videos in order using OpenCV (no audio).

Features:
- Accept multiple input videos; write a single output.
- Optional target FPS (resample per clip with frame duplication/drop to preserve duration).
- Optional target resolution (width/height) or scale factor; otherwise match the first video's size.
- Backend selection and fourcc fallback for better compatibility.

Examples:
  # Simple concat matching first video's size/FPS
  python concat_videos.py --videos a.mp4 b.mp4 c.mp4 --output out/merged.mp4 --backend ffmpeg --fourcc mp4v

  # Force 720p and 24fps
  python concat_videos.py --videos a.mp4 b.mp4 --output out/merged_720p.mp4 \
    --width 1280 --height 720 --fps 24 --backend ffmpeg --fourcc mp4v

Notes:
- Audio is NOT preserved.
- If H.264 is unavailable, use mp4v/XVID/MJPG or re-encode inputs.
"""

import argparse
import os
import sys
from typing import List, Sequence, Tuple

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


def _choose_interpolation(src_wh: Tuple[int, int], dst_wh: Tuple[int, int]) -> int:
    sw, sh = src_wh
    dw, dh = dst_wh
    if dw < sw or dh < sh:
        return cv2.INTER_AREA  # downscale
    return cv2.INTER_LINEAR   # upscale / same


def _resize_if_needed(frame, target_size: Tuple[int, int], interp: int):
    h, w = frame.shape[:2]
    if (w, h) == target_size:
        return frame
    return cv2.resize(frame, target_size, interpolation=interp)


def concat_videos(
    video_paths: Sequence[str],
    output_path: str,
    target_fps: float | None,
    target_size: Tuple[int, int] | None,
    backend: str,
    fourcc: str,
    fallback_fourccs: Sequence[str],
    pad_last_frame: bool,
):
    if not video_paths:
        raise ValueError("No videos provided")

    # Determine reference size/fps if not provided
    first_cap = _open_video(video_paths[0], backend)
    ref_w = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_h = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_fps = first_cap.get(cv2.CAP_PROP_FPS)
    if not ref_fps or ref_fps <= 0:
        ref_fps = 25.0
    first_cap.release()

    out_w, out_h = target_size if target_size else (ref_w, ref_h)
    out_fps = target_fps if target_fps and target_fps > 0 else ref_fps

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = _make_writer(output_path, out_fps, (out_w, out_h), fourcc, fallback_fourccs)

    try:
        for path in video_paths:
            cap = _open_video(path, backend)
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            if not src_fps or src_fps <= 0:
                src_fps = ref_fps
            factor = out_fps / src_fps
            interp = _choose_interpolation(
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                (out_w, out_h),
            )
            acc = 0.0
            last_frame_resized = None
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    acc += factor
                    while acc >= 1.0:
                        out_frame = _resize_if_needed(frame, (out_w, out_h), interp)
                        writer.write(out_frame)
                        last_frame_resized = out_frame
                        acc -= 1.0
                if pad_last_frame and last_frame_resized is not None:
                    writer.write(last_frame_resized)
            finally:
                cap.release()
    finally:
        writer.release()


def parse_args():
    p = argparse.ArgumentParser(description="Concatenate videos sequentially (OpenCV only, no audio)")
    p.add_argument("--videos", nargs="+", required=True, help="Input videos in order")
    p.add_argument("--output", required=True, help="Output video path")
    p.add_argument("--fps", type=float, help="Target FPS; default is FPS of first video")
    size = p.add_mutually_exclusive_group()
    size.add_argument("--scale", type=float, help="Uniform scale factor applied to first-video size")
    size.add_argument("--width", type=int, help="Target width (must also provide --height)")
    p.add_argument("--height", type=int, help="Target height (required if --width is used)")
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
    p.add_argument(
        "--pad-last-frame",
        action="store_true",
        help="After each clip, copy its last frame once more into the output",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.width is not None and args.height is None:
        print("Error: --height is required when --width is provided", file=sys.stderr)
        sys.exit(1)
    if args.scale is not None and args.scale <= 0:
        print("Error: --scale must be positive", file=sys.stderr)
        sys.exit(1)

    target_size = None
    if args.width is not None and args.height is not None:
        target_size = (args.width, args.height)
    elif args.scale is not None:
        # Need first video size
        tmp_cap = _open_video(args.videos[0], args.backend)
        src_w = int(tmp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(tmp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tmp_cap.release()
        if src_w <= 0 or src_h <= 0:
            print("Error: failed to read source dimensions", file=sys.stderr)
            sys.exit(1)
        target_size = (max(1, int(round(src_w * args.scale))), max(1, int(round(src_h * args.scale))))

    fallbacks = [c.strip() for c in args.fallback_fourccs.split(",") if c.strip()]

    try:
        concat_videos(
            video_paths=args.videos,
            output_path=args.output,
            target_fps=args.fps,
            target_size=target_size,
            backend=args.backend,
            fourcc=args.fourcc,
            fallback_fourccs=fallbacks,
            pad_last_frame=args.pad_last_frame,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Wrote concatenated video -> {args.output}")


if __name__ == "__main__":
    main()
