#!/usr/bin/env python3
"""
Automated workflow: split videos, generate/select masks, run inference per chunk, concat results.

Uses existing scripts in the repo:
- `batch_video_process.py` to split input and background videos
- `generate_blank_masks.py` to produce blank mask PNG sequences (if needed)
- `inference.py` to run model per chunk
- `concat_videos.py` to merge final chunks

Example:
  python run_full_workflow.py --input ./assets/LIGHT/talk/talk.mp4 \
    --bg ./assets/LIGHT/talk/output_mask_barbiepink.mp4 \
    --out-dir ./output/workflow --prompt "change lamp light color to barbie pink"
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import cv2


def run_cmd(cmd: List[str], cwd: str | None = None):
    print("+ ", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def list_parts(dirpath: Path) -> List[Path]:
    if not dirpath.exists():
        return []
    parts = sorted([p for p in dirpath.iterdir() if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.mkv')])
    return parts


def count_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return cnt


def count_pngs(mask_dir: Path) -> int:
    if not mask_dir.exists() or not mask_dir.is_dir():
        return 0
    return len([p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() == '.png'])


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Source (foreground) video")
    p.add_argument("--bg", required=True, help="Background conditioning video")
    p.add_argument("--out-dir", required=True, help="Output folder for splits, masks, results and final video")
    p.add_argument("--split-seconds", type=float, default=2.0, help="Segment length (seconds), default 2.0")
    p.add_argument("--fps", type=float, default=8.0, help="Target FPS for segments, default 8")
    p.add_argument("--prompt", required=True, help="Text prompt for inference")
    p.add_argument("--config", default="configs/inference_fbc.yaml", help="Model config path for inference.py")
    p.add_argument("--backend", default="ffmpeg", help="VideoCapture backend to use for split/concat")
    p.add_argument("--fourcc", default="mp4v", help="FourCC for writers")
    p.add_argument("--fallback-fourccs", default="mp4v,XVID,MJPG")
    p.add_argument("--mask-root", default=None, help="Optional root folder to search for existing masks (per-chunk). If not provided, masks are generated into out-dir/masks.")
    args = p.parse_args()

    inp = Path(args.input)
    bg = Path(args.bg)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    splits_in = out_dir / "splits_input"
    splits_bg = out_dir / "splits_bg"
    masks_out = out_dir / "masks"
    results_out = out_dir / "results"
    ensure_dir(splits_in)
    ensure_dir(splits_bg)
    ensure_dir(masks_out)
    ensure_dir(results_out)

    fallback_csv = args.fallback_fourccs

    # 1) Split input video
    cmd_in = [
        sys.executable, "batch_video_process.py",
        "--videos", str(inp),
        "--out-dir", str(splits_in),
        "--split-seconds", str(args.split_seconds),
        "--fps", str(args.fps),
        "--backend", args.backend,
        "--fourcc", args.fourcc,
        "--fallback-fourccs", fallback_csv,
    ]
    run_cmd(cmd_in, cwd=str(Path(__file__).parent))

    # 2) Split background video
    cmd_bg = [
        sys.executable, "batch_video_process.py",
        "--videos", str(bg),
        "--out-dir", str(splits_bg),
        "--split-seconds", str(args.split_seconds),
        "--fps", str(args.fps),
        "--backend", args.backend,
        "--fourcc", args.fourcc,
        "--fallback-fourccs", fallback_csv,
    ]
    run_cmd(cmd_bg, cwd=str(Path(__file__).parent))

    # 3) Collect parts
    in_parts = list_parts(splits_in)
    bg_parts = list_parts(splits_bg)

    if not in_parts:
        print("No input parts produced; aborting", file=sys.stderr)
        sys.exit(1)
    if not bg_parts:
        print("No bg parts produced; aborting", file=sys.stderr)
        sys.exit(1)

    count = min(len(in_parts), len(bg_parts))
    if len(in_parts) != len(bg_parts):
        print(f"Warning: input parts={len(in_parts)} bg parts={len(bg_parts)}; pairing first {count} parts")

    # 4) For each part: ensure mask, run inference
    for idx in range(count):
        in_part = in_parts[idx]
        bg_part = bg_parts[idx]
        base = in_part.stem  # e.g., talk_part001

        # expected frames by querying video
        expected_frames = count_frames(in_part)
        if expected_frames <= 0:
            print(f"Failed to read frames for {in_part}; skipping", file=sys.stderr)
            continue

        # mask candidate: if user specified mask-root, look there for a folder named like the input part stem
        if args.mask_root:
            candidate = Path(args.mask_root) / base
        else:
            candidate = masks_out / base

        if count_pngs(candidate) == expected_frames:
            mask_dir = candidate
            print(f"Using existing mask: {mask_dir} (frames={expected_frames})")
        else:
            # generate blank masks for this part
            ensure_dir(candidate)
            cmd_mask = [
                sys.executable, "generate_blank_masks.py",
                "--video", str(in_part),
                "--out", str(candidate),
                "--value", "255",
                "--digits", "4",
            ]
            run_cmd(cmd_mask, cwd=str(Path(__file__).parent))
            mask_dir = candidate

        # output path for inference result
        result_path = results_out / f"{base}_out.mp4"

        # run inference.py for this chunk
        cmd_inf = [
            sys.executable, "inference.py",
            "--input", str(in_part),
            "--mask", str(mask_dir),
            "--bg_cond", str(bg_part),
            "--config_path", args.config,
            "--output_path", str(result_path),
            "--prompt", args.prompt
        ]
        run_cmd(cmd_inf, cwd=str(Path(__file__).parent))

    # 5) Collect result chunks and concat
    res_parts = sorted([p for p in results_out.iterdir() if p.is_file() and p.suffix == '.mp4'])
    if not res_parts:
        print("No result chunks produced; aborting", file=sys.stderr)
        sys.exit(1)

    final_out = out_dir / "final_merged.mp4"
    cmd_concat = [
        sys.executable, "concat_videos.py",
        "--videos",
    ] + [str(p) for p in res_parts] + [
        "--output", str(final_out),
        "--fps", str(args.fps),
        "--backend", args.backend,
        "--fourcc", args.fourcc,
        "--fallback-fourccs", fallback_csv,
        "--pad-last-frame"
    ]

    run_cmd(cmd_concat, cwd=str(Path(__file__).parent))

    print(f"Done. Final merged video: {final_out}")


if __name__ == "__main__":
    main()
