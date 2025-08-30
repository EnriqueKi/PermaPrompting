# image dithering script
# © 2022 Roel Roscam Abbing, released as AGPLv3
# see https://www.gnu.org/licenses/agpl-3.0.html
# Support your local low-tech magazine: https://solar.lowtechmagazine.com/donate.html 

import hitherdither
import os
import argparse
import shutil
from PIL import Image
import logging
from pathlib import Path

parser = argparse.ArgumentParser(
    """
    This script recursively traverses folders and creates dithered versions of the images it finds.
    These are stored in the same folder as the images in a folder called "dithers".
    """
)

parser.add_argument(
    '-d', '--directory', help="Input directory to traverse (deprecated alias of --in)", default=None
)
parser.add_argument(
    '-i', '--in', dest='in_dir', help="Input directory containing original images", default='.'
)
parser.add_argument(
    '-o', '--out', dest='out_dir', help="Output directory for dithered images (required)", required=True
)
parser.add_argument(
    '--flat', action='store_true', help="Do not recreate subdirectory structure; dump all outputs directly into output directory"
)
parser.add_argument(
    '--suffix', default='_dithered', help="Filename suffix before extension (use '' to disable)"
)
parser.add_argument(
    '--auto-palette', type=int, help='Derive a per-image palette with this many colors (overrides --colorize & default grayscale for that image).'
)
parser.add_argument(
    '--global-palette', type=int, help='Build one shared palette of this many colors for all images (ignored if --auto-palette for a specific image).'
)
parser.add_argument(
    '--global-sample-images', type=int, default=150, help='Number of images to sample (uniform spread) when building global palette.'
)

parser.add_argument(
    '-rm', '--remove', help="Removes all the folders with dithers and their contents", action="store_true" 
    )

parser.add_argument(
    '-c', '--colorize', help="Colorizes the dithered images", action="store_true" 
    )

parser.add_argument(
    '-v', '--verbose', help="Print out more detailed information about what this script is doing", action="store_true" 
    )

args = parser.parse_args()

image_ext = [".jpg", ".JPG", ".jpeg", ".png", ".gif", ".webp", ".tiff", ".bmp"]

# Resolve input directory preference
content_dir = args.directory if args.directory is not None else args.in_dir
input_root = Path(content_dir).resolve()
output_root = Path(args.out_dir).resolve()
output_root.mkdir(parents=True, exist_ok=True)

if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

exclude_dirs = set(["dithers"])


logging.info("Dithering all images in %s and writing results to %s", content_dir, output_root)
logging.debug("excluding directories: {}".format("".join(exclude_dirs)))

def build_global_palette(image_paths, colors, sample_images):
    if not image_paths:
        return hitherdither.palette.Palette([(25,25,25),(75,75,75),(125,125,125),(175,175,175),(225,225,225),(250,250,250)])
    # Uniformly sample image_paths
    paths = image_paths
    if len(paths) > sample_images:
        step = max(1, len(paths)//sample_images)
        paths = paths[::step][:sample_images]
    pixels = []
    target_pixels = colors * 800  # heuristic
    for p in paths:
        try:
            with Image.open(p) as im:
                im = im.convert('RGB')
                data = list(im.getdata())
                if not data:
                    continue
                stride = max(1, len(data)//5000)
                pixels.extend(data[::stride])
                if len(pixels) >= target_pixels:
                    break
        except Exception:
            continue
    if not pixels:
        return hitherdither.palette.Palette([(25,25,25),(75,75,75),(125,125,125),(175,175,175),(225,225,225),(250,250,250)])
    import math
    # Create a temporary 1-row image to quantize
    temp = Image.new('RGB', (len(pixels), 1))
    temp.putdata(pixels[:len(pixels)])
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
    if not uniq:
        uniq = [(25,25,25),(75,75,75),(125,125,125),(175,175,175),(225,225,225),(250,250,250)]
    return hitherdither.palette.Palette(uniq)

def colorize(source_image, category):
    """
    Picks a colored dithering palette based on the post category.
    """

    colors = {
            'low-tech': hitherdither.palette.Palette([(30,32,40), (11,21,71),(57,77,174),(158,168,218),(187,196,230),(243,244,250)]),
            'obsolete': hitherdither.palette.Palette([(9,74,58), (58,136,118),(101,163,148),(144,189,179),(169,204,195),(242,247,246)]),
            'high-tech': hitherdither.palette.Palette([(86,9,6), (197,49,45),(228,130,124),(233,155,151),(242,193,190),(252,241,240)]),
            'grayscale': hitherdither.palette.Palette([(25,25,25), (75,75,75),(125,125,125),(175,175,175),(225,225,225),(250,250,250)])
        }


    if category:

        for i in colors.keys():
            if i in category.lower():
                color = colors[i]
                logging.info("Applying color palette '{}' for {}".format(i, category))
                break
            else:
                logging.info("No category for {}, {}".format(source_image, category))
                print("No category for {}, {}".format(source_image, category))
                color = colors['grayscale']

    else:
        logging.info("No category for {}, {}".format(source_image, category))
        print("No category for {}, {}".format(source_image, category))
        color = colors['grayscale']
        
    return color


def dither_image(source_image, output_image, category ='grayscale'):
    #see hitherdither docs for different dithering algos and settings
    if args.auto_palette:
        # Build palette from the image itself (median cut via Pillow quantize)
        try:
            with Image.open(source_image) as im_src:
                im_small = im_src.convert('RGB')
                # speed: shrink largest dimension to 400 for palette extraction
                if max(im_small.size) > 400:
                    scale = 400 / max(im_small.size)
                    new_size = (int(im_small.width*scale), int(im_small.height*scale))
                    im_small = im_small.resize(new_size, resample=getattr(Image,'BILINEAR',2))
                q = im_small.quantize(colors=args.auto_palette, method=0, dither=0)  # type: ignore[arg-type]
                raw = q.getpalette() or []
                colors = []
                for i in range(0, len(raw), 3):
                    if raw[i:i+3]:
                        colors.append(tuple(raw[i:i+3]))
                    if len(colors) >= args.auto_palette:
                        break
                if not colors:
                    colors = [(25,25,25),(125,125,125),(250,250,250)]
                # remove duplicates while preserving order
                seen = set()
                uniq = []
                for c in colors:
                    if c not in seen:
                        seen.add(c)
                        uniq.append(c)
                palette = hitherdither.palette.Palette(uniq)
        except Exception as ap_e:
            logging.debug("Auto-palette failed for %s: %s", source_image, ap_e)
            palette = hitherdither.palette.Palette([(25,25,25), (75,75,75),(125,125,125),(175,175,175),(225,225,225),(250,250,250)])
    elif args.colorize:
        palette = colorize(source_image, category)
    elif GLOBAL_PALETTE_OBJ is not None:
        palette = GLOBAL_PALETTE_OBJ
    else:
        palette = hitherdither.palette.Palette([(25,25,25), (75,75,75),(125,125,125),(175,175,175),(225,225,225),(250,250,250)])

    try:
        img = Image.open(source_image).convert('RGB')
        lanczos = getattr(Image, 'LANCZOS', getattr(Image, 'ANTIALIAS', getattr(Image, 'BICUBIC', 3)))
        img.thumbnail((800,800), resample=lanczos)  # type: ignore[arg-type]
        threshold = [96, 96, 96]
        img_dithered = hitherdither.ordered.bayer.bayer_dithering(img, palette, threshold, order=8)
        img_dithered.save(output_image, optimize=True)
    except Exception as e:
        logging.debug("❌ failed to convert %s", source_image)
        logging.debug(e)

def delete_dithers(content_dir):
    logging.info("Deleting 'dither' folders in {} and below".format(content_dir))
    for root, dirs, files in os.walk(content_dir, topdown=True):
        if root.endswith('dithers'):
            shutil.rmtree(root)
            logging.info("Removed {}".format(root))
        

def parse_front_matter(md):
    with open(md) as f:
        contents = f.readlines()
        cat = None
        for l in contents:
            if l.startswith("categories: "):
                cat = l.split("categories: ")[1]
                cat = cat.strip("[")
                cat = cat.strip()
                cat = cat.strip("]")

                logging.debug("Categories: {} from {}".format(cat, l.strip()))
        return cat

prev_root = None

if args.remove:
    delete_dithers(
        os.path.abspath(content_dir)
        )
else:
    # Pre-list all image files to optionally build a global palette
    all_image_files = []
    for r, dnames, fnames in os.walk(os.path.abspath(content_dir), topdown=True):
        dnames[:] = [d for d in dnames if d not in exclude_dirs]
        for fn in fnames:
            if fn.endswith(tuple(image_ext)):
                all_image_files.append(os.path.join(r, fn))

    if args.global_palette and not args.auto_palette and not args.colorize:
        logging.info("Building global palette of %d colors (sampling up to %d images)", args.global_palette, args.global_sample_images)
        GLOBAL_PALETTE_OBJ = build_global_palette(all_image_files, args.global_palette, args.global_sample_images)
        logging.info("Global palette ready (%d colors)", len(getattr(GLOBAL_PALETTE_OBJ, 'colors', getattr(GLOBAL_PALETTE_OBJ,'palette', []))))
    else:
        GLOBAL_PALETTE_OBJ = None

    for root, dirs, files in os.walk(os.path.abspath(content_dir), topdown=True):
        logging.debug("Checking next folder {}".format(root))

        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        category = None
        if prev_root is None:
            prev_root = root

    # (Removed legacy per-folder 'dithers' directory creation; using centralized out_dir now)

        if args.colorize:
            #iterate over md files to find one with a category
            if not category:
                for i in os.listdir(root):
                    if i.startswith('index'):
                        category2 = parse_front_matter(os.path.join(root,i))
                        
                        break


        for fname in files:
            if not fname.endswith(tuple(image_ext)):
                continue
            file_, ext = os.path.splitext(fname)
            source_image = os.path.join(root, fname)
            if args.flat:
                out_dir = output_root
            else:
                rel_dir = os.path.relpath(root, content_dir)
                out_dir = output_root / rel_dir
            os.makedirs(out_dir, exist_ok=True)
            suffix = args.suffix
            out_name = f"{file_}{suffix}{'.png' if suffix or ext.lower() != '.png' else ext}" if suffix else f"{file_}.png"
            output_image = str(out_dir / out_name)
            if not os.path.exists(output_image):
                if not args.colorize:
                    category2 = "grayscale"
                else:
                    category2 = category2 or "grayscale"
                dither_image(source_image, output_image, category2)
                logging.info("🖼 converted %s -> %s", fname, output_image)
            else:
                logging.debug("Dithered version exists, skipping %s", output_image)

        prev_root = root


logging.info("Done dithering")
