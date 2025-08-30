#!/usr/bin/env python3
"""
dither_videos.py — Dither / palette-reduce videos by extracting frames, applying ordered Bayer dithering
(similar to dither_images.py), and re-encoding.

Requirements:
  - ffmpeg (in PATH)
  - Python packages: Pillow, hitherdither
Optional (speed / UX): tqdm

Example usages:
  # Per-frame auto palette (24 colors), Floyd–Steinberg disabled (ordered Bayer) re-encode mp4
  python dither_videos.py --in input.mp4 --out output_dithered.mp4 --auto-palette 24

  # Global palette sampled from 200 frames (16 colors) for consistent look
  python dither_videos.py --in input.mov --out output_dithered.webm --global-palette 16 --global-sample-frames 200

  # Grayscale 6 levels
  python dither_videos.py --in input.mp4 --out output_bw.mp4 --grayscale-levels 6

    # Preset: match still-image 24-color ordered Bayer look (auto palette per frame, width 800, RGB output)
    python dither_videos.py --in input.mp4 --out output_preset.mp4 --preset-image24

    # Preset: small (size-focused) -> 16 colors, width 640, standard H.264 4:2:0, higher CRF, optional fps cap
    python dither_videos.py --in input.mp4 --out output_small.mp4 --preset-small

Notes:
  - Output inherits input FPS & audio unless --no-audio set.
  - Frames are processed in parallel (CPU cores) unless --no-parallel.
  - Temporary frame folders are auto-removed unless --keep-temp.
  - Ordered Bayer dithering uses hitherdither 8x8 matrix; can switch to none or (future) FS.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from PIL import Image
except Exception:
    print("Pillow required. pip install Pillow", file=sys.stderr)
    raise
HAS_HITHERDITHER = True
try:
    import hitherdither  # type: ignore
except Exception:  # Fallback implementation
    HAS_HITHERDITHER = False
    print("[info] hitherdither not available; using internal ordered Bayer fallback", file=sys.stderr)

    class _SimplePalette:
        def __init__(self, colors):
            # Ensure list of (r,g,b)
            self.colors = [tuple(map(int, c)) for c in colors if c is not None]

    # 8x8 Bayer matrix
    _BAYER_8 = [
        [0,48,12,60,3,51,15,63],
        [32,16,44,28,35,19,47,31],
        [8,56,4,52,11,59,7,55],
        [40,24,36,20,43,27,39,23],
        [2,50,14,62,1,49,13,61],
        [34,18,46,30,33,17,45,29],
        [10,58,6,54,9,57,5,53],
        [42,26,38,22,41,25,37,21],
    ]

    import numpy as _np

    def _ordered_bayer_dither(im: 'Image.Image', palette: '_SimplePalette', strength: float = 48.0):
        arr = _np.array(im.convert('RGB'), dtype=_np.float32)
        h, w, _ = arr.shape
        bayer = _np.array(_BAYER_8, dtype=_np.float32)
        # Normalize to [-0.5,0.5]
        bayer_n = (bayer + 0.5) / 64.0 - 0.5
        # Tile to image size
        tile = _np.tile(bayer_n, ( (h +7)//8, (w+7)//8))[:h, :w]
        # Apply to luminance-ish each channel equally
        arr += tile[..., None] * strength
        _np.clip(arr, 0, 255, out=arr)
        pal = _np.array(palette.colors, dtype=_np.float32)
        if pal.size == 0:
            pal = _np.array([(0,0,0),(255,255,255)], dtype=_np.float32)
        # Reshape pixels list
        pixels = arr.reshape(-1,3)
        # Compute squared distance to palette (broadcast)
        # (N,3) vs (K,3) -> (N,K)
        dists = _np.sum((pixels[:,None,:] - pal[None,:,:])**2, axis=2)
        idx = _np.argmin(dists, axis=1)
        mapped = pal[idx].astype(_np.uint8)
        out = mapped.reshape(h,w,3)
        return Image.fromarray(out, 'RGB')

# Optional progress
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

VALID_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def run(cmd: List[str], quiet=False):
    if not quiet:
        print("+", " ".join(cmd), file=sys.stderr)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR: {res.stderr.decode('utf-8','ignore')[:1000]}")
    return res


def extract_frames(video: Path, frames_dir: Path, max_width: Optional[int], fps: Optional[float]):
    """Extract frames, optionally resizing & reducing FPS.

    Ensures even dimensions (pad) so that subsequent encodes with subsampled
    pixel formats like yuv420p (libx264 default) don't fail with
    "height not divisible by 2" (or width) errors.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    scale_filter = (
        f"scale='min({max_width},iw)':-1:force_original_aspect_ratio=decrease"
        if max_width else "scale=iw:ih"
    )
    # Always pad to even dimensions; harmless if already even
    pad_filter = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    fps_part = f",fps={fps}" if fps else ""
    # Chain: scale -> (optional fps) -> pad
    vf = f"{scale_filter}{fps_part},{pad_filter}"
    out_pattern = str(frames_dir / "%06d.png")
    run(["ffmpeg", "-hide_banner", "-y", "-i", str(video), "-vf", vf, out_pattern])


def extract_audio(video: Path, audio_path: Path):
    run(["ffmpeg", "-hide_banner", "-y", "-i", str(video), "-vn", "-acodec", "copy", str(audio_path)])


def get_fps(video: Path) -> float:
    # Use ffprobe
    try:
        res = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", str(video)], quiet=True)
        txt = res.stdout.decode().strip()
        if "/" in txt:
            num, den = txt.split("/")
            return float(num) / float(den)
        return float(txt)
    except Exception:
        return 25.0  # fallback


def build_grayscale_palette(levels: int):
    if levels < 2:
        levels = 2
    steps = [int(round(i * 255 / (levels - 1))) for i in range(levels)]
    cols = [(v, v, v) for v in steps]
    if HAS_HITHERDITHER:
        return hitherdither.palette.Palette(cols)  # type: ignore[attr-defined]
    return _SimplePalette(cols)


def auto_palette_for_image(img: Image.Image, colors: int):
    im_small = img
    if max(im_small.size) > 400:
        scale = 400 / max(im_small.size)
        im_small = im_small.resize((int(im_small.width * scale), int(im_small.height * scale)), resample=getattr(Image, 'BILINEAR', 2))
    q = im_small.quantize(colors=colors, method=0, dither=0)  # type: ignore[arg-type]
    raw = q.getpalette() or []
    uniq = []
    seen = set()
    for i in range(0, len(raw), 3):
        if len(uniq) >= colors:
            break
        trip = tuple(raw[i:i+3])
        if len(trip) != 3:
            continue
        if trip not in seen:
            seen.add(trip)
            uniq.append(trip)
    if not uniq:
        uniq = [(25,25,25),(125,125,125),(250,250,250)]
    if HAS_HITHERDITHER:
        return hitherdither.palette.Palette(uniq)  # type: ignore[attr-defined]
    return _SimplePalette(uniq)


def build_global_palette(sample_frames: List[Path], colors: int):
    samples: List[Tuple[int,int,int]] = []
    target = colors * 800  # heuristic: gather many pixels
    for f in sample_frames:
        if len(samples) >= target:
            break
        try:
            with Image.open(f) as im:
                im = im.convert("RGB")
                data = list(im.getdata())
                if not data:
                    continue
                stride = max(1, len(data) // 5000)
                samples.extend(data[::stride])
        except Exception:
            continue
    if not samples:
        return build_grayscale_palette(colors if colors < 256 else 256)
    temp = Image.new("RGB", (len(samples), 1))
    temp.putdata(samples[:len(samples)])
    q = temp.quantize(colors=colors, method=0, dither=0)  # type: ignore[arg-type]
    raw = q.getpalette() or []
    uniq = []
    seen = set()
    for i in range(0, len(raw), 3):
        if len(uniq) >= colors:
            break
        trip = tuple(raw[i:i+3])
        if len(trip) != 3:
            continue
        if trip not in seen:
            seen.add(trip)
            uniq.append(trip)
    if HAS_HITHERDITHER:
        return hitherdither.palette.Palette(uniq)  # type: ignore[attr-defined]
    return _SimplePalette(uniq)


def dither_frame(src: Path, dst: Path, palette, use_auto: bool, auto_colors: int, grayscale_levels: Optional[int]):
    with Image.open(src) as im:
        im = im.convert("RGB")
        if grayscale_levels is not None:
            palette_use = build_grayscale_palette(grayscale_levels)
        elif use_auto:
            palette_use = auto_palette_for_image(im, auto_colors)
        else:
            palette_use = palette
        if HAS_HITHERDITHER:
            threshold = [96,96,96]
            d = hitherdither.ordered.bayer.bayer_dithering(im, palette_use, threshold, order=8)  # type: ignore[attr-defined]
        else:
            # Normalize palette object difference
            if not isinstance(palette_use, _SimplePalette):
                try:
                    cols = getattr(palette_use, 'colors', None) or getattr(palette_use, 'palette', None)
                    if cols is None:
                        raise ValueError
                    palette_use = _SimplePalette(cols)
                except Exception:
                    palette_use = _SimplePalette([(0,0,0),(255,255,255)])
            d = _ordered_bayer_dither(im, palette_use)
        d.save(dst, optimize=True)


def assemble_video(frames_dir: Path, fps: float, out_path: Path, audio_path: Optional[Path], vcodec: str, crf: int, pix_fmt: str):
    pattern = str(frames_dir / "%06d.png")
    cmd = ["ffmpeg", "-hide_banner", "-y", "-framerate", f"{fps}", "-i", pattern]
    if audio_path and audio_path.exists():
        cmd += ["-i", str(audio_path)]
        # Audio codec decision is deferred; replaced later if re-encode flags present (string replacement hack avoided by constructing final list later)
        # Placeholder will be adjusted by caller when invoking assemble_video if needed.
    cmd += ["-c:v", vcodec]
    if vcodec in ("libx264", "libx264rgb", "libx265"):
        cmd += ["-crf", str(crf), "-pix_fmt", pix_fmt]
    elif vcodec == "libvpx-vp9":
        cmd += ["-b:v", "0", "-crf", str(crf)]
    cmd.append(str(out_path))
    run(cmd)


def main():
    ap = argparse.ArgumentParser(description="Dither / palette-reduce a video via frame extraction.")
    ap.add_argument("--in", dest="inp", help="Input video file (omit when using --in-dir)")
    # Batch directory mode (alternative to --in)
    ap.add_argument("--in-dir", help="Input directory containing video files (alternative to --in). If set, --out-dir required.")
    ap.add_argument("--out-dir", help="Output directory for processed videos when using --in-dir")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories with --in-dir")
    ap.add_argument("--batch-suffix", default="_dithered", help="Suffix appended to stem for batch outputs (default: _dithered)")
    ap.add_argument("--out", dest="outp", help="Output video file (mp4/webm/mkv) (omit when using --out-dir)")
    ap.add_argument("--max-width", type=int, help="Resize width (preserve aspect) before processing")
    ap.add_argument("--fps", type=float, help="Override frame rate (frames per second)")
    ap.add_argument("--global-palette", type=int, help="Size of shared palette across all frames")
    ap.add_argument("--global-sample-frames", type=int, default=150, help="Number of frames to sample for global palette (uniformly spread)")
    ap.add_argument("--auto-palette", type=int, help="Per-frame auto-palette size (overrides global if set)")
    ap.add_argument("--grayscale-levels", type=int, help="Force grayscale with N levels (overrides palettes)")
    ap.add_argument("--no-audio", action="store_true", help="Drop audio track")
    ap.add_argument("--no-parallel", action="store_true", help="Disable multiprocessing (debug)")
    ap.add_argument("--keep-temp", action="store_true", help="Keep temporary frames directory")
    ap.add_argument("--vcodec", default="libx264", choices=["libx264","libx264rgb","libx265","libvpx-vp9"], help="Video codec")
    ap.add_argument("--crf", type=int, default=28, help="Quality CRF (lower=better) for applicable codecs")
    ap.add_argument("--pix-fmt", default="yuv420p", help="Pixel format (yuv420p default for broad compatibility)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--preset-image24", action="store_true", help="24-color ordered Bayer look (auto palette per frame, width 800, libx264rgb, rgb24, CRF 20)")
    ap.add_argument("--preset-small", action="store_true", help="Size-focused: auto palette 16, width 640, libx264 yuv420p, CRF 32, fps capped at 20 if higher")
    ap.add_argument("--audio-bitrate", type=int, help="Re-encode audio at this kbps instead of copying (e.g. 32)")
    ap.add_argument("--audio-mono", action="store_true", help="Downmix audio to mono when re-encoding")
    ap.add_argument("--audio-samplerate", type=int, help="Resample audio to this Hz when re-encoding (e.g. 16000)")
    args = ap.parse_args()

    # Validate mutually exclusive modes
    if args.in_dir:
        if args.inp and args.inp != '':
            print("Use either --in (single file) or --in-dir (batch), not both", file=sys.stderr)
            sys.exit(2)
        if not args.out_dir:
            print("--out-dir required when using --in-dir", file=sys.stderr)
            sys.exit(2)
        in_dir_path = Path(args.in_dir).resolve()
        if not in_dir_path.is_dir():
            print(f"Input directory not found: {in_dir_path}", file=sys.stderr)
            sys.exit(1)
        out_dir_path = Path(args.out_dir).resolve()
        out_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        if not args.inp or not args.outp:
            print("--in and --out required for single-file mode", file=sys.stderr)
            sys.exit(2)
        vin = Path(args.inp).resolve()
        if not vin.exists():
            print(f"Input not found: {vin}", file=sys.stderr)
            sys.exit(1)

    # Guard against incompatible presets (in single-file; batch will re-apply per file)
    if args.preset_image24 and args.preset_small:
        print("Cannot combine --preset-image24 and --preset-small", file=sys.stderr)
        sys.exit(2)

    # Apply preset adjustments before deriving FPS / processing
    if getattr(args, 'preset_image24', False):
        if args.max_width is None:
            args.max_width = 800
        if args.auto_palette is None:
            args.auto_palette = 24
        # Disable conflicting palette modes
        args.global_palette = None
        args.grayscale_levels = None
        # Upgrade codec / pixel format if still defaults
        if args.vcodec == 'libx264':
            args.vcodec = 'libx264rgb'
        if args.pix_fmt == 'yuv420p':
            args.pix_fmt = 'rgb24'
        if args.crf == 28:  # only change if user left default
            args.crf = 20
    elif getattr(args, 'preset_small', False):
        if args.max_width is None:
            args.max_width = 640
        if args.auto_palette is None:
            args.auto_palette = 16
        # Disable conflicting palette modes
        args.global_palette = None
        args.grayscale_levels = None
        # Ensure we use standard subsampled format for compression
        if args.vcodec == 'libx264rgb':
            args.vcodec = 'libx264'
        if args.pix_fmt == 'yuv420p':
            pass  # already optimal
        else:
            # Only force if user left default; if user explicitly changed keep it
            # (We cannot distinguish easily, so we only force when default) -> handled above
            pass
        if args.crf == 28:
            args.crf = 32  # higher CRF = smaller
        # Mark for later FPS capping
        _cap_small_fps = True
    else:
        _cap_small_fps = False

    def process_single_video(vpath: Path, out_path: Path):
        # Clone dynamic values which may be adjusted per video
        # We shallow-copy relevant args into a lightweight object (simple namespace)
        from types import SimpleNamespace
        a = SimpleNamespace(**vars(args))
        # Re-apply preset tweaks per video (in case batch)
        if a.preset_image24 and a.preset_small:
            print("Cannot combine --preset-image24 and --preset-small", file=sys.stderr)
            return
        if a.preset_image24:
            if a.max_width is None:
                a.max_width = 800
            if a.auto_palette is None:
                a.auto_palette = 24
            a.global_palette = None
            a.grayscale_levels = None
            if a.vcodec == 'libx264':
                a.vcodec = 'libx264rgb'
            if a.pix_fmt == 'yuv420p':
                a.pix_fmt = 'rgb24'
            if a.crf == 28:
                a.crf = 20
            cap_small = False
        elif a.preset_small:
            if a.max_width is None:
                a.max_width = 640
            if a.auto_palette is None:
                a.auto_palette = 16
            a.global_palette = None
            a.grayscale_levels = None
            if a.vcodec == 'libx264rgb':
                a.vcodec = 'libx264'
            if a.crf == 28:
                a.crf = 32
            cap_small = True
        else:
            cap_small = False

        fps_local = a.fps or get_fps(vpath)
        if cap_small and a.fps is None and fps_local > 20:
            if a.verbose:
                print(f"[info] Capping FPS from {fps_local:.2f} to 20 for --preset-small", file=sys.stderr)
            fps_local = 20.0

        temp_root = Path(tempfile.mkdtemp(prefix="dither_video_"))
        frames_dir = temp_root / "frames"
        out_frames_dir = temp_root / "frames_proc"
        out_frames_dir.mkdir(parents=True, exist_ok=True)
        audio_path = temp_root / "audio.mka"
        if a.verbose:
            print(f"[info] Processing {vpath.name}", file=sys.stderr)
        try:
            # Use capped / effective fps_local for extraction so we don't throw frames away later inconsistently
            extract_frames(vpath, frames_dir, a.max_width, fps_local if a.fps is None else a.fps)
            if not a.no_audio:
                try:
                    extract_audio(vpath, audio_path)
                except Exception as ae:
                    if a.verbose:
                        print(f"[warn] audio extract failed: {ae}", file=sys.stderr)
                    audio_path = None
            else:
                audio_path = None

            frame_files = sorted(frames_dir.glob('*.png'))
            if not frame_files:
                raise RuntimeError("No frames extracted")

            global_palette_obj = None
            if a.global_palette and not a.auto_palette and a.grayscale_levels is None:
                total = len(frame_files)
                step = max(1, total // max(1, a.global_sample_frames))
                sample_subset = frame_files[::step][:a.global_sample_frames]
                if a.verbose:
                    print(f"[info] Building global palette of {a.global_palette} colors from {len(sample_subset)} sampled frames", file=sys.stderr)
                global_palette_obj = build_global_palette(sample_subset, a.global_palette)

            grayscale_levels = a.grayscale_levels
            if grayscale_levels is not None:
                gray_pal = build_grayscale_palette(grayscale_levels)
            else:
                gray_pal = None

            def proc_frame(src: Path):
                dst = out_frames_dir / src.name
                try:
                    if grayscale_levels is not None:
                        dither_frame(src, dst, gray_pal, False, 0, grayscale_levels)
                    elif a.auto_palette:
                        dither_frame(src, dst, global_palette_obj or gray_pal, True, a.auto_palette, None)
                    elif global_palette_obj is not None:
                        dither_frame(src, dst, global_palette_obj, False, 0, None)
                    else:
                        dither_frame(src, dst, None, True, 12, None)
                except Exception as e:
                    if a.verbose:
                        print(f"[warn] frame {src.name} failed: {e}", file=sys.stderr)
                    shutil.copy(src, dst)

            if a.no_parallel:
                for f in tqdm(frame_files, disable=not a.verbose):
                    proc_frame(f)
            else:
                try:
                    import multiprocessing as mp
                    with mp.Pool() as pool:
                        list(tqdm(pool.imap_unordered(lambda p: (proc_frame(p), None)[0], frame_files), total=len(frame_files), disable=not a.verbose))
                except Exception as pe:
                    if a.verbose:
                        print(f"[warn] parallel failed ({pe}); falling back to sequential", file=sys.stderr)
                    for f in tqdm(frame_files, disable=not a.verbose):
                        proc_frame(f)

            # Assemble
            pattern = str(out_frames_dir / "%06d.png")
            final_out = out_path.resolve()
            cmd = ["ffmpeg", "-hide_banner", "-y", "-framerate", f"{fps_local}", "-i", pattern]
            if audio_path and audio_path.exists() and not a.no_audio:
                cmd += ["-i", str(audio_path)]
            cmd += ["-c:v", a.vcodec]
            if a.vcodec in ("libx264","libx264rgb","libx265"):
                cmd += ["-crf", str(a.crf), "-pix_fmt", a.pix_fmt]
            elif a.vcodec == "libvpx-vp9":
                cmd += ["-b:v", "0", "-crf", str(a.crf)]
            if audio_path and audio_path.exists() and not a.no_audio:
                if a.audio_bitrate or a.audio_mono or a.audio_samplerate:
                    ab = f"{a.audio_bitrate}k" if a.audio_bitrate else "32k"
                    ext = final_out.suffix.lower()
                    acodec = "libopus" if ext in (".webm", ".mkv") else "aac"
                    cmd += ["-c:a", acodec, "-b:a", ab]
                    if a.audio_mono:
                        cmd += ["-ac", "1"]
                    if a.audio_samplerate:
                        cmd += ["-ar", str(a.audio_samplerate)]
                else:
                    cmd += ["-c:a", "copy"]
            if a.verbose:
                print("+"," ".join(cmd), file=sys.stderr)
            # Append output file (was missing, causing "At least one output file must be specified")
            cmd.append(str(final_out))
            run(cmd)
            if a.verbose:
                print(f"[info] Wrote {final_out}", file=sys.stderr)
        finally:
            if not a.keep_temp:
                shutil.rmtree(temp_root, ignore_errors=True)
            elif a.verbose:
                print(f"[info] Kept temp dir {temp_root}", file=sys.stderr)

    # Batch mode
    if args.in_dir:
        # Collect video files
        if args.recursive:
            files = [p for p in in_dir_path.rglob('*') if p.is_file() and p.suffix.lower() in VALID_VIDEO_EXT]
        else:
            files = [p for p in in_dir_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_VIDEO_EXT]
        if not files:
            print(f"No video files found in {in_dir_path}", file=sys.stderr)
            sys.exit(1)
        for vp in files:
            stem = vp.stem + args.batch_suffix
            out_file = out_dir_path / (stem + vp.suffix)
            try:
                process_single_video(vp, out_file)
            except Exception as e:
                print(f"[error] Failed {vp.name}: {e}", file=sys.stderr)
        return

    # Single file path continues below
    fps = args.fps or get_fps(vin)

    # Cap FPS after reading if small preset requested and no explicit fps
    if '_cap_small_fps' in locals() and _cap_small_fps and args.fps is None and fps > 20:
        if args.verbose:
            print(f"[info] Capping FPS from {fps:.2f} to 20 for --preset-small", file=sys.stderr)
        fps = 20.0

    temp_root = Path(tempfile.mkdtemp(prefix="dither_video_"))
    frames_dir = temp_root / "frames"
    out_frames_dir = temp_root / "frames_proc"
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    audio_path = temp_root / "audio.mka"

    try:
        # Use possibly capped fps (variable 'fps') for extraction so timing matches output
        extract_frames(vin, frames_dir, args.max_width, fps)
        if not args.no_audio:
            try:
                extract_audio(vin, audio_path)
            except Exception as ae:
                if args.verbose:
                    print(f"[warn] audio extract failed: {ae}", file=sys.stderr)
                audio_path = None
        else:
            audio_path = None

        frame_files = sorted(frames_dir.glob('*.png'))
        if not frame_files:
            raise RuntimeError("No frames extracted")

        # Build global palette if requested and not using per-frame auto
        global_palette_obj = None
        if args.global_palette and not args.auto_palette and args.grayscale_levels is None:
            total = len(frame_files)
            step = max(1, total // max(1, args.global_sample_frames))
            sample_subset = frame_files[::step][:args.global_sample_frames]
            if args.verbose:
                print(f"[info] Building global palette of {args.global_palette} colors from {len(sample_subset)} sampled frames", file=sys.stderr)
            global_palette_obj = build_global_palette(sample_subset, args.global_palette)

        # Pre-build grayscale palette if used
        grayscale_levels = args.grayscale_levels
        if grayscale_levels is not None:
            gray_pal = build_grayscale_palette(grayscale_levels)
        else:
            gray_pal = None

        # Worker
        def process_one(src: Path):
            dst = out_frames_dir / src.name
            try:
                if grayscale_levels is not None:
                    dither_frame(src, dst, gray_pal, False, 0, grayscale_levels)
                elif args.auto_palette:
                    dither_frame(src, dst, global_palette_obj or gray_pal, True, args.auto_palette, None)
                elif global_palette_obj is not None:
                    dither_frame(src, dst, global_palette_obj, False, 0, None)
                else:
                    # fallback: per-frame 12-color auto palette
                    dither_frame(src, dst, None, True, 12, None)
            except Exception as e:
                if args.verbose:
                    print(f"[warn] frame {src.name} failed: {e}", file=sys.stderr)
                # copy original frame to maintain sync
                shutil.copy(src, dst)

        if args.no_parallel:
            for f in tqdm(frame_files, disable=not args.verbose):
                process_one(f)
        else:
            try:
                import multiprocessing as mp
                with mp.Pool() as pool:
                    list(tqdm(pool.imap_unordered(lambda p: (process_one(p), None)[0], frame_files), total=len(frame_files), disable=not args.verbose))
            except Exception as pe:
                if args.verbose:
                    print(f"[warn] parallel failed ({pe}); falling back to sequential", file=sys.stderr)
                for f in tqdm(frame_files, disable=not args.verbose):
                    process_one(f)

        # Assemble video
        if args.verbose:
            print("[info] Assembling video", file=sys.stderr)
        # Build base command via assemble_video then append / modify audio flags if re-encoding requested.
        # Simpler: reconstruct command here instead of modifying inside assemble_video for clarity.
        pattern = str(out_frames_dir / "%06d.png")
        final_out = Path(args.outp).resolve()
        cmd = ["ffmpeg", "-hide_banner", "-y", "-framerate", f"{fps}", "-i", pattern]
        if audio_path and audio_path.exists() and not args.no_audio:
            cmd += ["-i", str(audio_path)]
        cmd += ["-c:v", args.vcodec]
        if args.vcodec in ("libx264","libx264rgb","libx265"):
            cmd += ["-crf", str(args.crf), "-pix_fmt", args.pix_fmt]
        elif args.vcodec == "libvpx-vp9":
            cmd += ["-b:v", "0", "-crf", str(args.crf)]
        # Audio handling
        if audio_path and audio_path.exists() and not args.no_audio:
            if args.audio_bitrate or args.audio_mono or args.audio_samplerate:
                # Re-encode
                ab = f"{args.audio_bitrate}k" if args.audio_bitrate else "32k"
                # Choose codec based on container
                ext = final_out.suffix.lower()
                if ext in (".webm", ".mkv"):
                    acodec = "libopus"
                else:
                    acodec = "aac"
                cmd += ["-c:a", acodec, "-b:a", ab]
                if args.audio_mono:
                    cmd += ["-ac", "1"]
                if args.audio_samplerate:
                    cmd += ["-ar", str(args.audio_samplerate)]
            else:
                cmd += ["-c:a", "copy"]
        else:
            # No audio requested; skip
            pass
        cmd.append(str(final_out))
        if args.verbose:
            print("[info] Assembling video", file=sys.stderr)
            print("+"," ".join(cmd), file=sys.stderr)
        run(cmd)
        if args.verbose:
            print("[info] Done", file=sys.stderr)
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)
        elif args.verbose:
            print(f"[info] Kept temp dir {temp_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
