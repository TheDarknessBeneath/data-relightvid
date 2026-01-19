import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def clip_image(x, min=0., max=1.):
    """Clamp image tensor to [min, max]."""
    return torch.clamp(x, min=min, max=max)

def unnormalize(x):
    """Convert tensor from [-1, 1] to [0, 1]."""
    return (x + 1) / 2

def load_and_process_video(video_path, num_frames=16, crop_size=1280):
    """Load and process the first num_frames of a video."""
    frames = []
    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(Image.fromarray(frame)))

    cap.release()

    if len(frames) < num_frames:
        raise ValueError(f"Insufficient frames: expected {num_frames}, got {len(frames)}.")

    return torch.stack(frames)

def save_video_from_frames(image_pred, save_pth, fps=8):
    """Save tensor frames as a video."""
    frames = [
        (clip_image(unnormalize(image_pred[0][i].unsqueeze(0))) * 255)
        .squeeze(0).detach().cpu().permute(1, 2, 0).numpy().astype("uint8")
        for i in range(image_pred.shape[1])
    ]
    video_clip = ImageSequenceClip(frames, fps=fps)
    video_clip.write_videofile(save_pth, codec='libx264')
    print(f"Video saved to {save_pth}")

def apply_mask_to_video(video_path, mask_folder, crop_size=1280):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # Convert [0, 1] to [-1, 1]
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask_path = os.path.join(mask_folder, f"{frame_idx:04d}.png")
        if not os.path.exists(mask_path):
            frame_idx += 1
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            frame_idx += 1
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Fix broadcasting issue
        # result_frame = np.where(mask[:, :, None] > 0, frame, 128).astype(np.uint8)
        mask_float = mask.astype(np.float32) / 255.0

        if mask_float.ndim == 2:
            mask_float = mask_float[:, :, None]

        gray_background = np.ones_like(frame, dtype=np.float32) * 128.0
        result_frame = frame.astype(np.float32) * mask_float + gray_background * (1.0 - mask_float)

        result_frame = np.clip(result_frame, 0, 255).astype(np.uint8)
        result_image = Image.fromarray(result_frame)
        processed_frame = transform(result_image)
        frames.append(processed_frame)
        frame_idx += 1

    cap.release()

    if not frames:
        raise ValueError("No valid frames found.")

    return torch.stack(frames)
