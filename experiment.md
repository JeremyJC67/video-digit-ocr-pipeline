# Experiment Log

## Qwen Multi-Reader Refinement
- **Goal**: reliably read the bottom controller temperature and the top monitor BC/O pressure from `dataset/IMG_6920.MOV`.
- **Pipeline**:
  1. Use `qwen_multi_reader.py` with `Qwen/Qwen3-VL-2B-Thinking` to sample frames at 1 FPS and collect the model’s free-form analysis (`raw` field).
  2. Feed each `raw` text into a secondary text-only model (`Qwen/Qwen2.5-3B-Instruct`) that must return a strict one-line JSON object.  
     Command (smoke test on 5 frames):
     ```bash
     source ~/miniconda/bin/activate video-digit-ocr
     PYTHONNOUSERSITE=1 python qwen_multi_reader.py \
       --video dataset/IMG_6920.MOV \
       --extract_fps 1 \
       --device cuda:0 \
       --dtype float16 \
       --max_new_tokens 512 \
       --refine_model Qwen/Qwen2.5-3B-Instruct \
       --refine_device cuda:0 \
       --refine_max_new_tokens 512 \
       --out_dir outputs_multi/img6920_refined \
       --limit 5
     ```
  3. Apply regex fallbacks + forward/backward smoothing so that BC/O remains at the physically reasonable value (~4.33e-4 mbar) even when a single frame is misread.
- **Result**: `outputs_multi/img6920_refined/results_multi.csv|jsonl` — all five sampled frames report `measured_temp_c = -132.1` and `bc_o_mbar = 0.000433`, while `raw` retains the original reasoning text for debugging.
