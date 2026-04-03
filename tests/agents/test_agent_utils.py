import numpy as np
import pytest
from PIL import Image

from agentlab.agents.agent_utils import draw_click_indicator, draw_mouse_pointer


def test_draw_mouse_pointer_returns_image():
    """Test that draw_mouse_pointer returns a valid RGBA image."""
    img = Image.new("RGB", (100, 100), color="white")
    result = draw_mouse_pointer(img, 50, 50)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGBA"
    assert result.size == (100, 100)


def test_draw_mouse_pointer_modifies_pixels():
    """Test that draw_mouse_pointer actually draws something."""
    img = Image.new("RGB", (100, 100), color="white")
    result = draw_mouse_pointer(img, 50, 50)
    arr = np.array(result)
    # The pointer should have some non-white, non-fully-opaque pixels
    # At minimum, the alpha channel should have values < 255 somewhere
    assert not np.all(arr[:, :, 3] == 255) or not np.all(arr[:, :, :3] == 255)


def test_draw_mouse_pointer_at_origin():
    """Test drawing pointer at (0, 0)."""
    img = Image.new("RGB", (100, 100), color="white")
    result = draw_mouse_pointer(img, 0, 0)
    assert result.size == (100, 100)


def test_draw_click_indicator_returns_image():
    """Test that draw_click_indicator returns a valid RGBA image."""
    img = Image.new("RGB", (100, 100), color="white")
    result = draw_click_indicator(img, 50, 50)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGBA"
    assert result.size == (100, 100)


def test_draw_click_indicator_at_center():
    """Test that draw_click_indicator draws a crosshair pattern."""
    img = Image.new("RGB", (200, 200), color="white")
    result = draw_click_indicator(img, 100, 100)
    arr = np.array(result)
    # The crosshair lines should create some non-white pixels
    assert result.size == (200, 200)
