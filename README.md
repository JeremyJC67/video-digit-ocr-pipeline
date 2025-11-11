# Video Digit OCR Pipeline

Mini-pipeline for sampling a laboratory control video at 1 FPS, detecting the temperature displayed on a blue LCD/LED screen, and reviewing the results interactively.

## 1. Install prerequisites

1. Install system Tesseract (required by `pytesseract`).
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`
2. Create a virtual environment and install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Optional: install `easyocr` (`pip install easyocr`) to enable the fallback OCR engine.

## 2. Run the video processor

```bash
python process_video.py --video input.mp4 --out_csv results.csv --annotated_mp4 annotated.mp4 --fps 1
```

What happens:
- First run opens an OpenCV window so you can drag a bounding box over the LCD/LED screen. Reuse that ROI later with `--roi X Y W H` to skip the picker.
- The video is sampled at the requested FPS (default 1). For each timestamp the pipeline crops the ROI, preprocesses it (bilateral filter + CLAHE + Otsu), runs Tesseract (EasyOCR fallback), and parses numeric values like `22.8`.
- Each sampled frame is written to `./frames/<HH-MM-SS>.jpg` and a row is appended to the CSV with `video_time`, `detected_number`, `confidence`, and the frame filename.
- If `--annotated_mp4` is provided, per-digit boxes are overlaid on each frame (translated from ROI coordinates) and encoded at the sampling FPS.

Useful flags:
- `--roi X Y W H` – skip the GUI picker if you already know the screen coordinates.
- `--engine_easy_first` – try EasyOCR before Tesseract.
- `--strict_unit` – only accept readings when `°C`/`C` is present in the OCR text.
- `--start_time 15` – skip the first N seconds before sampling.
- `--max_duration 60` – cap processing time (seconds).
- `--max_percent 25` – only analyze the first X percent of the clip.
- `--frames_dir ./frames_custom` – choose where sampled images are stored.

## 3. Review and correct results

After processing, launch the Streamlit UI to audit low-confidence rows. The review app sorts rows by ascending confidence and lets you edit the detected numbers while viewing their frames.

```bash
streamlit run review_app.py -- --csv results.csv --frames_dir ./frames
```

Use the slider to highlight rows below a chosen confidence threshold. When you are done editing, click **Save final_results.csv** to export the corrected table next to the input CSV.

## 4. Notes & recommendations

- Keep your ROI tight around the digits—the preprocessing assumes a mostly blue/bright display.
- Lighting glare varies per clip; the bilateral filter + CLAHE combo keeps digits legible even under moderate blur.
- For automated QA, consider re-running `process_video.py` with `--strict_unit` so rows without units are left blank (these show up visibly in the review UI).
- The CSV plus frame dumps are designed to be version-controlled, making it easy to re-run OCR on improved models later.
