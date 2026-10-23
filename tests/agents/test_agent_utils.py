from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw

from agentlab.agents.agent_utils import (
    draw_arrowhead,
    draw_click_indicator,
    draw_mouse_pointer,
    zoom_webpage,
)


@pytest.fixture
def white_image():
    return Image.new("RGB", (100, 100), color=(255, 255, 255))


def test_draw_mouse_pointer_returns_new_rgba_image(white_image):
    result = draw_mouse_pointer(white_image, 10, 10)

    assert result is not white_image
    assert result.mode == "RGBA"
    assert result.size == white_image.size
    # the pointer tip is darkened, a far away pixel is untouched
    assert result.getpixel((12, 11))[:3] != (255, 255, 255)
    assert result.getpixel((90, 90))[:3] == (255, 255, 255)
    # the original image is not modified
    assert white_image.getpixel((12, 11)) == (255, 255, 255)


def test_draw_click_indicator_draws_plus_with_gap(white_image):
    result = draw_click_indicator(white_image, 50, 50)

    assert result.mode == "RGBA"
    # center is left blank (gap), arms are drawn
    assert result.getpixel((50, 50))[:3] == (255, 255, 255)
    assert result.getpixel((50, 40))[:3] == (0, 0, 0)
    assert result.getpixel((50, 60))[:3] == (0, 0, 0)
    assert result.getpixel((40, 50))[:3] == (0, 0, 0)
    assert result.getpixel((60, 50))[:3] == (0, 0, 0)
    assert result.getpixel((5, 5))[:3] == (255, 255, 255)


def test_draw_arrowhead_draws_two_lines():
    draw = MagicMock(spec=ImageDraw.ImageDraw)
    draw_arrowhead(draw, (0, 0), (50, 0))

    assert draw.line.call_count == 2
    for call in draw.line.call_args_list:
        points = call.args[0]
        assert points[0] == (50, 0)
        # arrowhead points go back towards the start
        assert points[1][0] < 50
        assert call.kwargs["fill"] == "red"


def test_zoom_webpage_sets_zoom():
    page = MagicMock()
    assert zoom_webpage(page, 2) is page
    page.evaluate.assert_called_once_with("document.documentElement.style.zoom='200%'")


@pytest.mark.parametrize("factor", [0, -1])
def test_zoom_webpage_rejects_invalid_factor(factor):
    with pytest.raises(ValueError):
        zoom_webpage(MagicMock(), factor)
