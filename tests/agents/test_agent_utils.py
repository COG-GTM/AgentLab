from math import cos, radians, sin
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image, ImageDraw

from agentlab.agents.agent_utils import (
    draw_arrowhead,
    draw_click_indicator,
    draw_mouse_pointer,
    zoom_webpage,
)


def _white(size=(100, 100), mode="RGB"):
    color = (255, 255, 255) if mode == "RGB" else (255, 255, 255, 255)
    return Image.new(mode, size, color)


def _changed_mask(before: Image.Image, after: Image.Image) -> np.ndarray:
    a = np.asarray(before.convert("RGBA")).astype(int)
    b = np.asarray(after.convert("RGBA")).astype(int)
    return np.any(a != b, axis=-1)


class TestDrawMousePointer:
    @pytest.mark.parametrize("mode", ["RGB", "RGBA"])
    def test_returns_new_rgba_image_of_same_size(self, mode):
        img = _white(mode=mode)
        out = draw_mouse_pointer(img, 10, 10)
        assert out is not img
        assert out.size == img.size
        assert out.mode == "RGBA"

    def test_does_not_mutate_input(self):
        img = _white()
        before = np.asarray(img).copy()
        draw_mouse_pointer(img, 30, 30)
        np.testing.assert_array_equal(np.asarray(img), before)

    def test_pixels_change_only_inside_pointer_box(self):
        x, y = 30, 40
        img = _white()
        out = draw_mouse_pointer(img, x, y)
        mask = _changed_mask(img, out)
        ys, xs = np.nonzero(mask)
        assert mask.sum() > 0
        # pointer polygon spans [x, x+20] x [y, y+20]
        assert xs.min() >= x and xs.max() <= x + 20
        assert ys.min() >= y and ys.max() <= y + 20

    def test_pointer_is_semi_transparent_black(self):
        x, y = 30, 40
        out = draw_mouse_pointer(_white(), x, y)
        # a point well inside the arrow polygon
        r, g, b, a = out.getpixel((x + 12, y + 10))
        assert a == 255
        assert r == g == b
        assert 100 <= r <= 155  # ~50% blend of black over white

    def test_far_away_pixels_untouched(self):
        out = draw_mouse_pointer(_white(), 5, 5)
        assert out.getpixel((90, 90)) == (255, 255, 255, 255)


class TestDrawClickIndicator:
    def test_returns_new_rgba_image_of_same_size(self):
        img = _white((80, 60))
        out = draw_click_indicator(img, 40, 30)
        assert out is not img
        assert out.size == (80, 60)
        assert out.mode == "RGBA"

    def test_center_gap_is_left_untouched(self):
        img = Image.new("RGB", (100, 100), (100, 150, 200))
        out = draw_click_indicator(img, 50, 50)
        assert out.getpixel((50, 50)) == (100, 150, 200, 255)

    @pytest.mark.parametrize("dx,dy", [(0, -9), (0, 9), (-9, 0), (9, 0)])
    def test_four_arms_are_black(self, dx, dy):
        img = Image.new("RGB", (100, 100), (100, 150, 200))
        out = draw_click_indicator(img, 50, 50)
        assert out.getpixel((50 + dx, 50 + dy)) == (0, 0, 0, 255)

    def test_changes_confined_to_cross_extent(self):
        img = Image.new("RGB", (100, 100), (100, 150, 200))
        out = draw_click_indicator(img, 50, 50)
        ys, xs = np.nonzero(_changed_mask(img, out))
        reach = 4 + 10 + 2  # gap + line_length + outline slack
        assert xs.min() >= 50 - reach and xs.max() <= 50 + reach
        assert ys.min() >= 50 - reach and ys.max() <= 50 + reach

    def test_does_not_mutate_input(self):
        img = _white()
        before = np.asarray(img).copy()
        draw_click_indicator(img, 50, 50)
        np.testing.assert_array_equal(np.asarray(img), before)


class TestDrawArrowhead:
    def test_draws_two_red_lines_from_end_with_expected_geometry(self):
        draw = MagicMock()
        draw_arrowhead(draw, (0, 0), (100, 0), arrow_length=15, arrow_angle=30)
        assert draw.line.call_count == 2
        (left_pts,), left_kw = draw.line.call_args_list[0]
        (right_pts,), right_kw = draw.line.call_args_list[1]
        assert left_pts[0] == (100, 0) and right_pts[0] == (100, 0)
        exp_x = 100 - 15 * cos(radians(30))
        assert left_pts[1] == pytest.approx((exp_x, 15 * sin(radians(30))))
        assert right_pts[1] == pytest.approx((exp_x, -15 * sin(radians(30))))
        assert left_kw == {"fill": "red", "width": 4}
        assert right_kw == {"fill": "red", "width": 4}

    def test_vertical_arrow_head_points_back_toward_start(self):
        draw = MagicMock()
        draw_arrowhead(draw, (50, 0), (50, 80))
        for call in draw.line.call_args_list:
            (_, tip) = call.args[0]
            assert tip[1] < 80  # wings trail behind the arrow tip

    def test_draws_red_pixels_on_real_image(self):
        img = _white()
        draw_arrowhead(ImageDraw.Draw(img), (10, 50), (80, 50))
        arr = np.asarray(img)
        red = (arr[..., 0] == 255) & (arr[..., 1] == 0) & (arr[..., 2] == 0)
        assert red.any()
        assert not red[:, :60].any()  # head is near the end point only


class TestZoomWebpage:
    def test_default_zoom(self):
        page = MagicMock()
        assert zoom_webpage(page) is page
        page.evaluate.assert_called_once_with("document.documentElement.style.zoom='150.0%'")

    @pytest.mark.parametrize("factor,expected", [(0.5, "50.0%"), (2, "200%"), (1.25, "125.0%")])
    def test_custom_zoom(self, factor, expected):
        page = MagicMock()
        zoom_webpage(page, factor)
        page.evaluate.assert_called_once_with(f"document.documentElement.style.zoom='{expected}'")

    @pytest.mark.parametrize("factor", [0, -1, -0.5])
    def test_non_positive_zoom_raises(self, factor):
        page = MagicMock()
        with pytest.raises(ValueError, match="greater than 0"):
            zoom_webpage(page, factor)
        page.evaluate.assert_not_called()
