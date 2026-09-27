"""Slide image utilities: discover images, read dimensions."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # whole-slide TIFs exceed Pillow's default bomb guard

from config import IMAGES_DIR, SLIDE_IMAGE_EXTS, TILE_WIDTH_PX, TILE_HEIGHT_PX


def find_slide_images(images_dir: Path = IMAGES_DIR) -> dict[str, Path]:
    """Return {slide_id: path} for all supported images in images_dir."""
    result: dict[str, Path] = {}
    for ext in SLIDE_IMAGE_EXTS:
        for p in images_dir.glob(f"*{ext}"):
            result[p.stem] = p
        for p in images_dir.glob(f"*{ext.upper()}"):
            result[p.stem] = p
    return result


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Return (width, height) in pixels without loading pixel data."""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def tile_grid_shape(image_w: int, image_h: int) -> tuple[int, int]:
    """Number of tiles in x and y directions (ceiling division)."""
    n_x = (image_w + TILE_WIDTH_PX - 1) // TILE_WIDTH_PX
    n_y = (image_h + TILE_HEIGHT_PX - 1) // TILE_HEIGHT_PX
    return n_x, n_y
