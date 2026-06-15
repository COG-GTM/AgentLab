from dataclasses import asdict

from browsergym.experiments.agent import AgentInfo
from PIL import Image, ImageDraw
from playwright.sync_api import Page


def busted_retry_ans_dict(max_retry: int) -> dict:
    """Fallback answer dict used when parsing the LLM response keeps failing.

    Args:
        max_retry: The maximum number of retries the agent allowed.

    Returns:
        An answer dict with a null action and the retry bookkeeping marking the
        step as a busted retry.
    """
    return dict(action=None, n_retry=max_retry + 1, busted_retry=1)


def make_agent_info(chat_llm, ans_dict: dict, chat_messages, chat_model_args) -> AgentInfo:
    """Build the ``AgentInfo`` returned by chat-based agents from ``get_action``.

    Collects the LLM call stats, augments them with the retry bookkeeping found
    in ``ans_dict`` and packages everything (thought, messages, stats, model
    args) into an ``AgentInfo``.

    Args:
        chat_llm: The chat model, used to retrieve per-call stats.
        ans_dict: The parsed answer dict (must contain ``n_retry`` and
            ``busted_retry``; ``think`` is optional).
        chat_messages: The messages exchanged with the model for this step.
        chat_model_args: The model args dataclass, serialized into ``extra_info``.

    Returns:
        The populated ``AgentInfo``.
    """
    stats = chat_llm.get_stats()
    stats["n_retry"] = ans_dict["n_retry"]
    stats["busted_retry"] = ans_dict["busted_retry"]

    return AgentInfo(
        think=ans_dict.get("think", None),
        chat_messages=chat_messages,
        stats=stats,
        extra_info={"chat_model_args": asdict(chat_model_args)},
    )


def draw_mouse_pointer(image: Image.Image, x: int, y: int) -> Image.Image:
    """
    Draws a semi-transparent mouse pointer at (x, y) on the image.
    Returns a new image with the pointer drawn.

    Args:
        image: The image to draw the mouse pointer on.
        x: The x coordinate for the mouse pointer.
        y: The y coordinate for the mouse pointer.

    Returns:
        A new image with the mouse pointer drawn.
    """
    pointer_size = 20  # Length of the pointer
    overlay = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(overlay)

    # Define pointer shape (a simple arrow)
    pointer_shape = [
        (x, y),
        (x + pointer_size, y + pointer_size // 2),
        (x + pointer_size // 2, y + pointer_size // 2),
        (x + pointer_size // 2, y + pointer_size),
    ]

    draw.polygon(pointer_shape, fill=(0, 0, 0, 128))  # 50% transparent black

    return Image.alpha_composite(image.convert("RGBA"), overlay)


def draw_arrowhead(draw, start, end, arrow_length=15, arrow_angle=30):
    from math import atan2, cos, radians, sin

    angle = atan2(end[1] - start[1], end[0] - start[0])
    left = (
        end[0] - arrow_length * cos(angle - radians(arrow_angle)),
        end[1] - arrow_length * sin(angle - radians(arrow_angle)),
    )
    right = (
        end[0] - arrow_length * cos(angle + radians(arrow_angle)),
        end[1] - arrow_length * sin(angle + radians(arrow_angle)),
    )
    draw.line([end, left], fill="red", width=4)
    draw.line([end, right], fill="red", width=4)


def draw_click_indicator(image: Image.Image, x: int, y: int) -> Image.Image:
    """
    Draws a click indicator (+ shape with disconnected lines) at (x, y) on the image.
    Returns a new image with the click indicator drawn.

    Args:
        image: The image to draw the click indicator on.
        x: The x coordinate for the click indicator.
        y: The y coordinate for the click indicator.

    Returns:
        A new image with the click indicator drawn.
    """
    line_length = 10  # Length of each line segment
    gap = 4  # Gap from center point
    line_width = 2  # Thickness of lines

    overlay = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(overlay)

    # Draw 4 lines forming a + shape with gaps in the center
    # Each line has a white outline and black center for visibility on any background

    # Top line
    draw.line(
        [(x, y - gap - line_length), (x, y - gap)], fill=(255, 255, 255, 200), width=line_width + 2
    )  # White outline
    draw.line(
        [(x, y - gap - line_length), (x, y - gap)], fill=(0, 0, 0, 255), width=line_width
    )  # Black center

    # Bottom line
    draw.line(
        [(x, y + gap), (x, y + gap + line_length)], fill=(255, 255, 255, 200), width=line_width + 2
    )  # White outline
    draw.line(
        [(x, y + gap), (x, y + gap + line_length)], fill=(0, 0, 0, 255), width=line_width
    )  # Black center

    # Left line
    draw.line(
        [(x - gap - line_length, y), (x - gap, y)], fill=(255, 255, 255, 200), width=line_width + 2
    )  # White outline
    draw.line(
        [(x - gap - line_length, y), (x - gap, y)], fill=(0, 0, 0, 255), width=line_width
    )  # Black center

    # Right line
    draw.line(
        [(x + gap, y), (x + gap + line_length, y)], fill=(255, 255, 255, 200), width=line_width + 2
    )  # White outline
    draw.line(
        [(x + gap, y), (x + gap + line_length, y)], fill=(0, 0, 0, 255), width=line_width
    )  # Black center

    return Image.alpha_composite(image.convert("RGBA"), overlay)


def zoom_webpage(page: Page, zoom_factor: float = 1.5):
    """
    Zooms the webpage to the specified zoom factor.

    NOTE: Click actions with bid doesn't work properly when zoomed in.

    Args:
        page: The Playwright Page object.
        zoom_factor: The zoom factor to apply (default is 1.5).

    Returns:
        Page: The modified Playwright Page object.

    Raises:
        ValueError: If zoom_factor is less than or equal to 0.
    """

    if zoom_factor <= 0:
        raise ValueError("Zoom factor must be greater than 0.")

    page.evaluate(f"document.documentElement.style.zoom='{zoom_factor*100}%'")
    return page
