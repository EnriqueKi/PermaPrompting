# PermaComputing Dithering Tools

Based on the work of Kris De Decker and Low Tech Magazine (https://github.com/lowtechmag/solar_v2). Utilities to batch dither / palette‑reduce images and videos in a "low‑tech-magazine" ordered Bayer style for size reduction and a consistent aesthetic. 

## Contents
- `dither_images.py` – Recursively process images, producing dithered PNGs
- `dither_videos.py` – Extract frames, apply ordered Bayer dithering, and reassemble video (with optional audio compression & presets)

## Quick Start
```bash
# (optional) create virtual environment
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install pillow numpy hitherdither tqdm
# ffmpeg must be installed (macOS: brew install ffmpeg)
```

## dither_images.py
Batch converts images inside an input directory to dithered PNGs.

### Basic per‑image auto palette (24 colors)
```bash
python dither_images.py --in Imgs/Original --out Imgs/Ditherd --auto-palette 24 --flat --suffix '' --verbose
```
- `--auto-palette N` – Build a palette from each image (median cut) with N colors.
- `--flat` – Do not reproduce folder tree; dump all outputs directly in the output directory.
- `--suffix ''` – No suffix; overwrite style (outputs always PNG). Use default `_dithered` to keep originals.
- `--colorize` – Use one of the preset thematic palettes if the category (from front matter) matches low-tech / obsolete / high-tech.

### Shared (global) palette
Ensures a consistent look across a whole set:
```bash
python dither_images.py --in Imgs/Original --out Imgs/Ditherd --global-palette 16 --flat --suffix '' --verbose
```
Options:
- `--global-palette N` – Build one palette of N colors across all images (sampled subset).
- `--global-sample-images M` – Limit how many images are sampled (default 150).

### Remove previously generated images
(Old mode that created per-folder `dithers` directories)
```bash
python dither_images.py --remove -i Imgs/Original -o Imgs/Ditherd
```

## dither_videos.py
Per‑frame dithering of videos via frame extraction → palette reduction → re-encode with ffmpeg.

### Typical auto‑palette per frame (24 colors)
```bash
python dither_videos.py --in input.mp4 --out output_dithered.mp4 --auto-palette 24 --max-width 800 --verbose
```

### Global palette (consistent colors across frames)
```bash
python dither_videos.py --in input.mov --out output_global.mp4 --global-palette 16 --global-sample-frames 200 --max-width 800
```

### Grayscale
```bash
python dither_videos.py --in input.mp4 --out output_bw.mp4 --grayscale-levels 6
```

### Presets
- `--preset-image24` – 24-color ordered Bayer, width 800, RGB mode (`libx264rgb`, `rgb24`, CRF 20)
- `--preset-small` – Size-focused: width 640, 16 colors auto palette, CRF 32, FPS capped at 20, standard `libx264 yuv420p`

Example:
```bash
python dither_videos.py --in input.mp4 --out tiny.mp4 --preset-small --audio-bitrate 32 --audio-mono --audio-samplerate 16000
```

### Audio options
- `--no-audio` – Drop audio entirely.
- `--audio-bitrate K` – Re-encode audio (AAC/Opus) at K kbps.
- `--audio-mono` – Downmix to mono (saves space).
- `--audio-samplerate Hz` – Resample (e.g. 16000).

### Batch video mode
Process all supported videos in a directory:
```bash
python dither_videos.py --in-dir Videos/Original --out-dir Videos/Dithered --global-palette 8 --max-width 320 --fps 12 --crf 36 --audio-bitrate 16 --audio-mono --audio-samplerate 12000 --verbose
```
Options:
- `--recursive` – Recurse into subdirectories.
- `--batch-suffix _dithered` – Custom suffix appended to each filename stem.

### Pixel formats & dimension safety
The script pads frames to even dimensions to satisfy codecs like H.264 (yuv420p) automatically.

### Parallelism
Both scripts can use multiprocessing (videos: per-frame). Use `--no-parallel` to disable for debugging.

## Dependency Summary
- Python: 3.9+ (tested with 3.11)
- Libraries: pillow, numpy, hitherdither, tqdm (optional progress)
- External: ffmpeg (including ffprobe) for video processing

If `hitherdither` is missing in video script a built-in ordered Bayer fallback is used; the image script currently requires `hitherdither`.

## Troubleshooting
- ImportError: hitherdither – Run `pip install hitherdither` in your active environment.
- ffmpeg not found – Install via package manager (macOS: `brew install ffmpeg`).
- Height not divisible by 2 – Already auto-padded; update script if using an older version.
- Large output size – Try `--preset-small` for videos or lower `--auto-palette` / `--global-palette` values (e.g. 12 or 8). Reduce width (`--max-width`). Add audio compression flags.

## Suggested Workflows
1. Generate a small, consistent color look for a photo set:
```bash
python dither_images.py --in Imgs/Original --out Imgs/Ditherd --global-palette 12 --flat --suffix ''
```
2. Produce a tiny shareable clip:
```bash
python dither_videos.py --in clip.mov --out clip_small.mp4 --preset-small --audio-bitrate 24 --audio-mono
```
3. Experimental palette tweaking: run with `--auto-palette 20`, inspect results, then lock a `--global-palette` using that count for consistency.

## License
Original image dithering script © 2022 Roel Roscam Abbing (AGPLv3). Enhancements and video tooling follow same license unless stated otherwise.

## Contribute
Feel free to open issues / suggestions for:
- Additional dithering algorithms (e.g. Floyd–Steinberg toggle)
- Adaptive palette reuse across cuts
- Animated GIF export helper

Enjoy the low‑tech aesthetic!
