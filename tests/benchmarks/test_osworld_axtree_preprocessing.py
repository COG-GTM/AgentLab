import io
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from agentlab.benchmarks.osworld_axtree_preprocessing import (
    draw_bounding_boxes,
    filter_nodes,
    find_leaf_nodes,
    judge_node,
    linearize_accessibility_tree,
    tag_screenshot,
    trim_accessibility_tree,
)

NS = (
    'xmlns:st="https://accessibility.ubuntu.example.org/ns/state" '
    'xmlns:cp="https://accessibility.ubuntu.example.org/ns/component" '
    'xmlns:attr="https://accessibility.windows.example.org/ns/attributes"'
)
VISIBLE = 'st:showing="true" st:visible="true" st:enabled="true"'

UBUNTU_TREE = f"""<desktop-frame {NS}>
  <panel>
    <push-button name="OK" {VISIBLE} cp:screencoord="(10, 10)" cp:size="(20, 20)" attr:class="btn" attr:description="confirm">OK</push-button>
    <label name="Hidden" st:showing="false" st:visible="true" st:enabled="true" cp:screencoord="(0, 0)" cp:size="(5, 5)"/>
  </panel>
  <push-button name="Offscreen" {VISIBLE} cp:screencoord="(-1, 5)" cp:size="(10, 10)"/>
  <push-button name="ZeroSize" {VISIBLE} cp:screencoord="(5, 5)" cp:size="(0, 10)"/>
  <entry name="Quote" {VISIBLE} cp:screencoord="(40, 40)" cp:size="(30, 10)">say "hi"</entry>
  <panel name="NotKeptTag" {VISIBLE} cp:screencoord="(1, 1)" cp:size="(1, 1)"/>
  <image {VISIBLE} image="true" cp:screencoord="(60, 60)" cp:size="(20, 20)"/>
</desktop-frame>"""


@pytest.fixture
def root():
    return ET.fromstring(UBUNTU_TREE)


@pytest.fixture
def noisy_png():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_find_leaf_nodes_empty():
    assert find_leaf_nodes("") == []
    assert find_leaf_nodes(None) == []


def test_find_leaf_nodes():
    tags = [n.get("name") for n in find_leaf_nodes(UBUNTU_TREE)]
    assert tags == ["OK", "Hidden", "Offscreen", "ZeroSize", "Quote", "NotKeptTag", None]


def test_filter_nodes_ubuntu(root):
    names = [n.get("name") for n in filter_nodes(root)]
    assert names == ["OK", "Quote"]


def test_filter_nodes_check_image(root):
    nodes = filter_nodes(root, check_image=True)
    assert [n.tag for n in nodes] == ["push-button", "entry", "image"]


def test_judge_node_invalid_platform(root):
    with pytest.raises(ValueError, match="Invalid platform"):
        judge_node(root, platform="macos")


def test_judge_node_windows_only_requires_visible():
    tree = (
        '<root xmlns:st="https://accessibility.windows.example.org/ns/state" '
        'xmlns:cp="https://accessibility.windows.example.org/ns/component">'
        '<button name="Win" st:visible="true" st:enabled="true" '
        'cp:screencoord="(1, 1)" cp:size="(5, 5)"/></root>'
    )
    node = ET.fromstring(tree)[0]
    assert judge_node(node, platform="windows")
    assert not judge_node(node, platform="ubuntu")  # ubuntu also needs "showing"


def test_linearize_accessibility_tree(root):
    lines = linearize_accessibility_tree(UBUNTU_TREE).split("\n")
    assert lines[0].startswith("tag\tname\ttext")
    assert lines[1] == "push-button\tOK\tOK\tbtn\tconfirm\t(10, 10)\t(20, 20)"
    assert lines[2] == 'entry\tQuote\t"say ""hi"""\t\t\t(40, 40)\t(30, 10)'
    assert len(lines) == 3


def test_linearize_invalid_platform():
    with pytest.raises(ValueError):
        linearize_accessibility_tree(UBUNTU_TREE, platform="bsd")


def test_draw_bounding_boxes(root, noisy_png):
    nodes = filter_nodes(root)
    marks, drew_nodes, text, image_bytes = draw_bounding_boxes(nodes, noisy_png)
    assert marks == [[10, 10, 20, 20], [40, 40, 30, 10]]
    assert [n.get("name") for n in drew_nodes] == ["OK", "Quote"]
    lines = text.split("\n")
    assert lines[0] == "index\ttag\tname\ttext"
    assert lines[1] == "1\tpush-button\tOK\tOK"
    assert lines[2] == '2\tentry\tQuote\t"say ""hi"""'
    assert Image.open(io.BytesIO(image_bytes)).size == (100, 100)


def test_draw_bounding_boxes_skips_uniform_regions(root):
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), (255, 255, 255)).save(buf, format="PNG")
    marks, drew_nodes, text, _ = draw_bounding_boxes(filter_nodes(root), buf.getvalue())
    assert marks == [] and drew_nodes == []
    assert text == "index\ttag\tname\ttext"


def test_draw_bounding_boxes_downsampling_keeps_original_marks(root, noisy_png):
    marks, _, _, image_bytes = draw_bounding_boxes(
        filter_nodes(root), noisy_png, down_sampling_ratio=0.5
    )
    assert marks[0] == [10, 10, 20, 20]
    assert Image.open(io.BytesIO(image_bytes)).size == (50, 50)


def test_draw_bounding_boxes_invalid_platform(noisy_png):
    with pytest.raises(ValueError):
        draw_bounding_boxes([], noisy_png, platform="plan9")


def test_tag_screenshot(noisy_png):
    marks, drew_nodes, tagged, element_list = tag_screenshot(noisy_png, UBUNTU_TREE)
    assert len(marks) == len(drew_nodes) == 3
    assert isinstance(tagged, bytes)
    assert element_list.split("\n")[3].startswith("3\timage")


def test_trim_accessibility_tree():
    short = "tag\tname"
    assert trim_accessibility_tree(short, max_tokens=100) == short
    long = "word " * 200
    trimmed = trim_accessibility_tree(long, max_tokens=10)
    assert trimmed.endswith("[...]\n")
    assert len(trimmed) < len(long)
