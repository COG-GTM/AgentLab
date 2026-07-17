from unittest.mock import MagicMock

import pytest
from PIL import Image

from agentlab.agents.agent_utils import (
    draw_arrowhead,
    draw_click_indicator,
    draw_mouse_pointer,
    zoom_webpage,
)


def test_draw_mouse_pointer_returns_new_image():
    image = Image.new("RGB", (100, 100), "white")
    result = draw_mouse_pointer(image, 50, 50)
    assert result is not image
    assert result.mode == "RGBA"
    assert result.size == (100, 100)
    # some pixels changed where the pointer was drawn
    assert result.getpixel((55, 55)) != (255, 255, 255, 255)


def test_draw_click_indicator_returns_new_image():
    image = Image.new("RGB", (100, 100), "red")
    result = draw_click_indicator(image, 50, 50)
    assert result.mode == "RGBA"
    assert result.size == (100, 100)
    # center pixel unchanged (gap), line pixels changed
    assert result.getpixel((50, 40)) != (255, 0, 0, 255)


def test_draw_arrowhead():
    image = Image.new("RGB", (100, 100), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    draw_arrowhead(draw, (10, 10), (60, 60))
    assert image.getpixel((59, 59)) != (255, 255, 255)


def test_zoom_webpage_calls_evaluate():
    page = MagicMock()
    result = zoom_webpage(page, 2.0)
    page.evaluate.assert_called_once()
    assert "200" in page.evaluate.call_args[0][0]
    assert result is page


def test_zoom_webpage_invalid_factor():
    with pytest.raises(ValueError):
        zoom_webpage(MagicMock(), 0)
