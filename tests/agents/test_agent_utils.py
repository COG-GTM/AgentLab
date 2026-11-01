from unittest.mock import MagicMock

import pytest
from PIL import Image

from agentlab.agents.agent_utils import (
    draw_arrowhead,
    draw_click_indicator,
    draw_mouse_pointer,
    zoom_webpage,
)


@pytest.fixture
def white_image():
    return Image.new("RGB", (100, 100), (255, 255, 255))


def test_draw_mouse_pointer_returns_new_rgba_image(white_image):
    out = draw_mouse_pointer(white_image, 10, 10)
    assert out is not white_image
    assert out.mode == "RGBA"
    assert out.size == white_image.size
    # pixel inside the arrow is darkened, pixel far away is untouched
    assert out.getpixel((15, 13))[:3] != (255, 255, 255)
    assert out.getpixel((90, 90))[:3] == (255, 255, 255)
    # original is not modified
    assert white_image.getpixel((15, 13)) == (255, 255, 255)


def test_draw_click_indicator_leaves_center_gap(white_image):
    out = draw_click_indicator(white_image, 50, 50)
    assert out.mode == "RGBA"
    assert out.getpixel((50, 50))[:3] == (255, 255, 255)  # gap at the center
    assert out.getpixel((50, 40))[:3] == (0, 0, 0)  # top line
    assert out.getpixel((60, 50))[:3] == (0, 0, 0)  # right line
    assert out.getpixel((5, 5))[:3] == (255, 255, 255)


def test_draw_arrowhead_draws_two_lines():
    draw = MagicMock()
    draw_arrowhead(draw, (0, 0), (100, 0))
    assert draw.line.call_count == 2
    for call in draw.line.call_args_list:
        points = call.args[0]
        assert points[0] == (100, 0)
        assert call.kwargs == {"fill": "red", "width": 4}


def test_zoom_webpage_evaluates_js():
    page = MagicMock()
    assert zoom_webpage(page, 2.0) is page
    page.evaluate.assert_called_once_with("document.documentElement.style.zoom='200.0%'")


@pytest.mark.parametrize("factor", [0, -1.5])
def test_zoom_webpage_invalid_factor(factor):
    page = MagicMock()
    with pytest.raises(ValueError):
        zoom_webpage(page, factor)
    page.evaluate.assert_not_called()
