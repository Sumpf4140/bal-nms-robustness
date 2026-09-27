"""Tests for core/slide_image.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from config import TILE_WIDTH_PX, TILE_HEIGHT_PX
from core.slide_image import get_image_dimensions, tile_grid_shape, find_slide_images


def make_image(path: Path, w: int, h: int) -> None:
    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    img.save(str(path))


class TestGetImageDimensions:
    def test_png(self, tmp_path):
        p = tmp_path / "slide.png"
        make_image(p, 1920, 1080)
        w, h = get_image_dimensions(p)
        assert w == 1920
        assert h == 1080

    def test_jpeg(self, tmp_path):
        p = tmp_path / "slide.jpg"
        make_image(p, 640, 480)
        w, h = get_image_dimensions(p)
        assert w == 640
        assert h == 480

    def test_tiff(self, tmp_path):
        p = tmp_path / "slide.tif"
        make_image(p, 3200, 2400)
        w, h = get_image_dimensions(p)
        assert w == 3200
        assert h == 2400

    def test_returns_tuple(self, tmp_path):
        p = tmp_path / "slide.png"
        make_image(p, 100, 200)
        result = get_image_dimensions(p)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestTileGridShape:
    def test_exact_multiple(self):
        nx, ny = tile_grid_shape(1920, 1280)
        assert nx == 1920 // TILE_WIDTH_PX
        assert ny == 1280 // TILE_HEIGHT_PX

    def test_non_exact_multiple_rounds_up(self):
        nx, ny = tile_grid_shape(641, 641)
        assert nx == 2
        assert ny == 2

    def test_single_tile(self):
        nx, ny = tile_grid_shape(640, 640)
        assert nx == 1
        assert ny == 1

    def test_smaller_than_tile(self):
        nx, ny = tile_grid_shape(100, 200)
        assert nx == 1
        assert ny == 1


class TestFindSlideImages:
    def test_finds_png(self, tmp_path):
        make_image(tmp_path / "slide001.png", 640, 640)
        slides = find_slide_images(tmp_path)
        assert "slide001" in slides

    def test_finds_multiple_extensions(self, tmp_path):
        make_image(tmp_path / "a.png", 100, 100)
        make_image(tmp_path / "b.jpg", 100, 100)
        make_image(tmp_path / "c.tif", 100, 100)
        slides = find_slide_images(tmp_path)
        assert "a" in slides
        assert "b" in slides
        assert "c" in slides

    def test_empty_directory(self, tmp_path):
        slides = find_slide_images(tmp_path)
        assert slides == {}

    def test_ignores_non_image_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("ignore me")
        slides = find_slide_images(tmp_path)
        assert "notes" not in slides

    def test_returns_paths(self, tmp_path):
        make_image(tmp_path / "slide.png", 100, 100)
        slides = find_slide_images(tmp_path)
        assert isinstance(slides["slide"], Path)
