import io
import sys
import types
import xml.etree.ElementTree as ET
from unittest.mock import patch

import pytest
from PIL import Image, ImageFont

from agentlab.benchmarks import osworld_axtree_preprocessing as pre

NS = {
    "ubuntu": {
        "st": pre.state_ns_ubuntu,
        "cp": pre.component_ns_ubuntu,
        "attr": pre.attributes_ns_ubuntu,
        "val": pre.value_ns_ubuntu,
    },
    "windows": {
        "st": pre.state_ns_windows,
        "cp": pre.component_ns_windows,
        "attr": pre.attributes_ns_windows,
        "val": pre.value_ns_windows,
    },
}


def _xmlns(platform):
    ns = NS[platform]
    return (
        f'xmlns:st="{ns["st"]}" xmlns:cp="{ns["cp"]}" xmlns:attr="{ns["attr"]}" '
        f'xmlns:val="{ns["val"]}" xmlns:cls="{pre.class_ns_windows}"'
    )


def _node_xml(
    tag="push-button",
    name="OK",
    text=None,
    platform="ubuntu",
    showing="true",
    visible="true",
    enabled="true",
    coord="(10, 10)",
    size="(20, 20)",
    extra="",
):
    attrs = []
    if name is not None:
        attrs.append(f'name="{name}"')
    if showing is not None:
        attrs.append(f'st:showing="{showing}"')
    if visible is not None:
        attrs.append(f'st:visible="{visible}"')
    if enabled is not None:
        attrs.append(f'st:enabled="{enabled}"')
    if coord is not None:
        attrs.append(f'cp:screencoord="{coord}"')
    if size is not None:
        attrs.append(f'cp:size="{size}"')
    if extra:
        attrs.append(extra)
    body = text if text is not None else ""
    return f"<{tag} {' '.join(attrs)}>{body}</{tag}>"


def _tree(*nodes, platform="ubuntu"):
    return f"<desktop-frame {_xmlns(platform)}>{''.join(nodes)}</desktop-frame>"


def _node(platform="ubuntu", **kwargs):
    root = ET.fromstring(_tree(_node_xml(platform=platform, **kwargs), platform=platform))
    return list(root)[0]


def _png(size=(100, 100), uniform=False, mode="RGB"):
    if uniform:
        img = Image.new(mode, size, "white")
    else:
        img = Image.linear_gradient("L").resize(size).convert(mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestFindLeafNodes:
    @pytest.mark.parametrize("empty", ["", None])
    def test_empty_input_returns_empty_list(self, empty):
        assert pre.find_leaf_nodes(empty) == []

    def test_collects_only_leaves_in_document_order(self):
        xml = "<a><b><c/><d/></b><e/><f><g><h/></g></f></a>"
        assert [n.tag for n in pre.find_leaf_nodes(xml)] == ["c", "d", "e", "h"]

    def test_single_root_is_a_leaf(self):
        assert [n.tag for n in pre.find_leaf_nodes("<root/>")] == ["root"]


class TestJudgeNode:
    @pytest.mark.parametrize(
        "tag",
        [
            "document-frame",
            "list-item",
            "push-button",
            "heading",
            "label",
            "scrollbar",
            "searchbox",
            "textbox",
            "link",
            "page-tabelement",
            "textfield",
            "textarea",
            "menu",
            "alert",
            "check-box",
            "entry",
            "paragraph",
            "table-cell",
            "terminal",
            "netuiribbontab",
            "uiribboncommandbar",
        ],
    )
    def test_interactive_tags_are_kept(self, tag):
        assert pre.judge_node(_node(tag=tag)) is True

    @pytest.mark.parametrize("tag", ["panel", "filler", "frame", "window", "menu-bar"])
    def test_non_interactive_tags_are_dropped(self, tag):
        assert pre.judge_node(_node(tag=tag)) is False

    def test_invalid_platform_raises(self):
        with pytest.raises(ValueError, match="Invalid platform"):
            pre.judge_node(_node(), platform="macos")

    @pytest.mark.parametrize(
        "showing,visible,expected",
        [("true", "true", True), ("false", "true", False), ("true", "false", False)],
    )
    def test_ubuntu_requires_showing_and_visible(self, showing, visible, expected):
        assert pre.judge_node(_node(showing=showing, visible=visible)) is expected

    def test_ubuntu_missing_state_attrs_default_to_false(self):
        assert pre.judge_node(_node(showing=None)) is False

    @pytest.mark.parametrize("visible,expected", [("true", True), ("false", False), (None, False)])
    def test_windows_only_requires_visible(self, visible, expected):
        node = _node(platform="windows", showing=None, visible=visible)
        assert pre.judge_node(node, platform="windows") is expected

    def test_platform_namespace_mismatch_drops_node(self):
        # windows state namespace is not read when judging as ubuntu
        node = _node(platform="windows")
        assert pre.judge_node(node, platform="ubuntu") is False

    @pytest.mark.parametrize("flag", ["editable", "expandable", "checkable"])
    def test_alternative_interactivity_flags_replace_enabled(self, flag):
        node = _node(enabled=None, extra=f'st:{flag}="true"')
        assert pre.judge_node(node) is True

    def test_not_enabled_and_no_other_flag_is_dropped(self):
        assert pre.judge_node(_node(enabled="false")) is False

    def test_text_content_substitutes_for_name(self):
        assert pre.judge_node(_node(name=None, text="hello")) is True

    def test_no_name_and_no_text_is_dropped(self):
        assert pre.judge_node(_node(name=None)) is False

    def test_image_flag_only_counts_with_check_image(self):
        node = _node(tag="image", name=None, extra='image="true"')
        assert pre.judge_node(node, check_image=False) is False
        assert pre.judge_node(node, check_image=True) is True

    @pytest.mark.parametrize(
        "coord,size,expected",
        [
            ("(0, 0)", "(1, 1)", True),
            ("(-1, 5)", "(10, 10)", False),
            ("(5, -1)", "(10, 10)", False),
            ("(5, 5)", "(0, 10)", False),
            ("(5, 5)", "(10, 0)", False),
            (None, "(10, 10)", False),
            ("(5, 5)", None, False),
        ],
    )
    def test_geometry_must_be_on_screen_and_non_empty(self, coord, size, expected):
        assert pre.judge_node(_node(coord=coord, size=size)) is expected


class TestFilterNodes:
    def test_returns_only_kept_nodes_in_iteration_order(self):
        xml = _tree(
            _node_xml(tag="push-button", name="A"),
            "<panel>" + _node_xml(tag="label", name="B") + "</panel>",
            _node_xml(tag="push-button", name="hidden", showing="false"),
            _node_xml(tag="filler", name="C"),
        )
        nodes = pre.filter_nodes(ET.fromstring(xml))
        assert [n.get("name") for n in nodes] == ["A", "B"]

    def test_check_image_is_forwarded(self):
        xml = _tree(_node_xml(tag="image", name=None, extra='image="true"'))
        root = ET.fromstring(xml)
        assert pre.filter_nodes(root) == []
        assert len(pre.filter_nodes(root, check_image=True)) == 1


class TestDrawBoundingBoxes:
    def _nodes(self, *xml_nodes, platform="ubuntu"):
        return list(ET.fromstring(_tree(*xml_nodes, platform=platform)))

    def test_invalid_platform_raises(self):
        with pytest.raises(ValueError, match="Invalid platform"):
            pre.draw_bounding_boxes([], _png(), platform="mac")

    @pytest.mark.parametrize("platform", ["ubuntu", "windows"])
    def test_draws_marks_and_builds_element_table(self, platform):
        nodes = self._nodes(
            _node_xml(tag="push-button", name="OK", coord="(10, 10)", size="(20, 20)"),
            _node_xml(tag="label", name="Title", text="Hello", coord="(40, 40)", size="(30, 10)"),
            platform=platform,
        )
        marks, drew, text, image_bytes = pre.draw_bounding_boxes(nodes, _png(), platform=platform)

        assert marks == [[10, 10, 20, 20], [40, 40, 30, 10]]
        assert [n.get("name") for n in drew] == ["OK", "Title"]
        assert text.splitlines() == [
            "index\ttag\tname\ttext",
            '1\tpush-button\tOK\t""',
            "2\tlabel\tTitle\tHello",
        ]
        out = Image.open(io.BytesIO(image_bytes))
        assert out.format == "PNG"
        assert out.size == (100, 100)

    def test_text_with_quotes_is_csv_escaped(self):
        nodes = self._nodes(_node_xml(name="Q", text='say "hi"'))
        _, _, text, _ = pre.draw_bounding_boxes(nodes, _png())
        assert text.splitlines()[1] == '1\tpush-button\tQ\t"say ""hi"""'

    @pytest.mark.parametrize(
        "value,expected",
        [("typed", "typed"), ('a "b"', '"a ""b"""')],
    )
    def test_edit_wrapper_value_used_when_no_text(self, value, expected):
        escaped = value.replace('"', "&quot;")
        nodes = self._nodes(
            _node_xml(
                tag="textbox",
                name="Edit",
                platform="windows",
                extra=f'cls:class="RichEditWrapper" val:value="{escaped}"',
            ),
            platform="windows",
        )
        _, _, text, _ = pre.draw_bounding_boxes(nodes, _png(), platform="windows")
        assert text.splitlines()[1] == f"1\ttextbox\tEdit\t{expected}"

    def test_edit_wrapper_without_value_falls_back_to_empty(self):
        nodes = self._nodes(
            _node_xml(tag="textbox", name="Edit", extra='cls:class="EditWrapper"'),
        )
        _, _, text, _ = pre.draw_bounding_boxes(nodes, _png())
        assert text.splitlines()[1] == '1\ttextbox\tEdit\t""'

    @pytest.mark.parametrize(
        "coord,size",
        [
            (None, "(10, 10)"),  # missing coordinates
            ("(10, 10)", None),  # missing size
            ("(a, b)", "(10, 10)"),  # unparsable -> ValueError swallowed
            ("(10, 10)", "(0, 10)"),  # non-positive size
            ("(10, 10)", "(10, -3)"),
        ],
    )
    def test_invalid_geometry_nodes_are_skipped(self, coord, size):
        nodes = self._nodes(_node_xml(coord=coord, size=size), _node_xml(name="Good"))
        marks, drew, text, _ = pre.draw_bounding_boxes(nodes, _png())
        assert [n.get("name") for n in drew] == ["Good"]
        assert marks == [[10, 10, 20, 20]]
        assert text.splitlines()[1].startswith("1\t")

    def test_single_colour_region_is_skipped(self):
        nodes = self._nodes(_node_xml())
        marks, drew, text, _ = pre.draw_bounding_boxes(nodes, _png(uniform=True))
        assert marks == [] and drew == []
        assert text == "index\ttag\tname\ttext"

    def test_down_sampling_resizes_image_but_marks_keep_original_geometry(self):
        nodes = self._nodes(_node_xml(coord="(20, 20)", size="(40, 40)"))
        marks, drew, _, image_bytes = pre.draw_bounding_boxes(
            nodes, _png((100, 100)), down_sampling_ratio=0.5
        )
        assert Image.open(io.BytesIO(image_bytes)).size == (50, 50)
        assert marks == [[20, 20, 40, 40]]

    def test_down_sampling_that_collapses_size_skips_node(self):
        nodes = self._nodes(_node_xml(coord="(20, 20)", size="(1, 1)"))
        marks, _, _, _ = pre.draw_bounding_boxes(nodes, _png(), down_sampling_ratio=0.5)
        assert marks == []

    def test_falls_back_to_default_font_when_truetype_missing(self):
        nodes = self._nodes(_node_xml())
        default = ImageFont.load_default()
        with (
            patch.object(pre.ImageFont, "truetype", side_effect=IOError("no font")),
            patch.object(pre.ImageFont, "load_default", return_value=default) as load_default,
        ):
            marks, _, _, _ = pre.draw_bounding_boxes(nodes, _png())
        load_default.assert_called_once_with()
        assert len(marks) == 1

    def test_uses_truetype_font_when_available(self):
        nodes = self._nodes(_node_xml())
        default = ImageFont.load_default()
        with patch.object(pre.ImageFont, "truetype", return_value=default) as truetype:
            marks, _, _, _ = pre.draw_bounding_boxes(nodes, _png())
        truetype.assert_called_once_with("arial.ttf", 15)
        assert len(marks) == 1


def test_print_nodes_with_indent_prints_tree(capsys):
    root = ET.fromstring('<a><b x="1"><c/></b></a>')
    pre.print_nodes_with_indent([root])
    lines = capsys.readouterr().out.splitlines()
    assert lines == [" a {}", "   b {'x': '1'}", "     c {}"]


class TestLinearizeAccessibilityTree:
    HEADER = "tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"

    def test_invalid_platform_raises(self):
        with pytest.raises(ValueError, match="Invalid platform"):
            pre.linearize_accessibility_tree(_tree(), platform="android")

    def test_ubuntu_rows(self):
        xml = _tree(
            _node_xml(
                tag="push-button",
                name="OK",
                extra='attr:class="GtkButton" attr:description="Confirm"',
            ),
            _node_xml(tag="label", name="L", text='He said "yo"', coord="(1, 2)", size="(3, 4)"),
            _node_xml(tag="filler", name="ignored"),
        )
        lines = pre.linearize_accessibility_tree(xml).splitlines()
        assert lines == [
            self.HEADER,
            'push-button\tOK\t""\tGtkButton\tConfirm\t(10, 10)\t(20, 20)',
            'label\tL\t"He said ""yo"""\t\t\t(1, 2)\t(3, 4)',
        ]

    def test_windows_rows_use_windows_class_namespace(self):
        xml = _tree(
            _node_xml(
                tag="textbox",
                name="Edit",
                platform="windows",
                showing=None,
                extra='cls:class="EditWrapper" val:value="x &quot;y&quot;"',
            ),
            _node_xml(
                tag="push-button",
                name="Plain",
                platform="windows",
                showing=None,
                extra='cls:class="Button"',
            ),
            platform="windows",
        )
        lines = pre.linearize_accessibility_tree(xml, platform="windows").splitlines()
        assert lines == [
            self.HEADER,
            'textbox\tEdit\t"x ""y"""\tEditWrapper\t\t(10, 10)\t(20, 20)',
            'push-button\tPlain\t""\tButton\t\t(10, 10)\t(20, 20)',
        ]

    def test_windows_edit_wrapper_plain_value(self):
        xml = _tree(
            _node_xml(
                tag="textbox",
                name="E",
                platform="windows",
                extra='cls:class="EditWrapper" val:value="abc"',
            ),
            platform="windows",
        )
        lines = pre.linearize_accessibility_tree(xml, platform="windows").splitlines()
        assert lines[1].split("\t")[2] == "abc"

    def test_empty_tree_returns_header_only(self):
        assert pre.linearize_accessibility_tree(_tree()) == self.HEADER


class TestTagScreenshot:
    def test_ubuntu_tags_visible_nodes_including_images(self):
        xml = _tree(
            _node_xml(name="OK"),
            _node_xml(tag="image", name=None, extra='image="true"', coord="(50, 50)"),
        )
        marks, drew, tagged, element_list = pre.tag_screenshot(_png(), xml)
        assert marks == [[10, 10, 20, 20], [50, 50, 20, 20]]
        assert len(drew) == 2
        assert Image.open(io.BytesIO(tagged)).format == "PNG"
        assert element_list.splitlines()[0] == "index\ttag\tname\ttext"
        assert len(element_list.splitlines()) == 3

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: tag_screenshot does not forward `platform` to draw_bounding_boxes, "
        "so windows trees are read with ubuntu namespaces and nothing is tagged",
    )
    def test_windows_platform_tags_nodes(self):
        xml = _tree(_node_xml(platform="windows", showing=None), platform="windows")
        marks, drew, _, _ = pre.tag_screenshot(_png(), xml, platform="windows")
        assert marks == [[10, 10, 20, 20]]
        assert len(drew) == 1


class TestTrimAccessibilityTree:
    @pytest.fixture
    def fake_tiktoken(self, monkeypatch):
        calls = []

        class _Enc:
            def encode(self, text):
                return list(text)

            def decode(self, tokens):
                return "".join(tokens)

        def encoding_for_model(model):
            calls.append(model)
            return _Enc()

        module = types.ModuleType("tiktoken")
        module.encoding_for_model = encoding_for_model
        monkeypatch.setitem(sys.modules, "tiktoken", module)
        return calls

    def test_short_text_is_returned_unchanged(self, fake_tiktoken):
        assert pre.trim_accessibility_tree("abc", max_tokens=3) == "abc"
        assert fake_tiktoken == ["gpt-4"]

    def test_long_text_is_truncated_with_marker(self, fake_tiktoken):
        assert pre.trim_accessibility_tree("abcdefgh", max_tokens=3) == "abc[...]\n"

    def test_zero_budget_keeps_only_marker(self, fake_tiktoken):
        assert pre.trim_accessibility_tree("abc", max_tokens=0) == "[...]\n"
