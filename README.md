### RelightVid
生成重光照结果：
以原视频为前景 `input`，重光照 mask 为背景 `bg`（脚本内部将重光照 mask 黑色部分处理为灰色，不修改前景中对应黑色重光照 mask 的部分），使用更改光照的 prompt 生成重光照视频
将原视频分为长 2s 的若干段，每段进行一次渲染，然后拼接各段视频得出结果

如果修改 fps 需要更改 inference.py 中 save_video_from_frames 函数参数，函数定义位于 RelightVid/utils/video_utils.py。
```bash
python run_full_workflow.py --input ./assets/LIGHT/talk/talk.mp4 \
  --bg ./assets/LIGHT/talk/output_mask_barbiepink.mp4 \
  --out-dir ./output/workflow --prompt "change lamp light color to barbie pink" \
  --fps 8 --split-seconds 2.0
```

模型路径：
```
RelightVid
├── models
│   ├── realistic-vision-v51                              // stable diffusion base model
│   │   ├── text_encoder
│   │   │   ├── config.json
│   │   │   └── model.safetensors
│   │   ├── tokenizer
│   │   │   ├── merges.txt
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.json
│   │   ├── unet
│   │   │   └── diffusion_pytorch_model.safetensors
│   │   ├── vae
│   │   │   ├── config.json
│   │   │   └── diffusion_pytorch_model.safetensors
│   ├── iclight_sd15_fbc.safetensors                      // ic-light weights
│   ├── relvid_mm_sd15_fbc.pth                            // relightvid motion weights
```