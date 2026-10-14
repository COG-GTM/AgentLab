import ast
import inspect
from types import SimpleNamespace

import pytest
from PIL import Image

from agentlab.analyze import overlay_utils
from agentlab.analyze.overlay_utils import ArgInfo


def _arg(name, value, start=0, stop=1, fn="f"):
    return ArgInfo(
        function_name=fn,
        name=name,
        value=value,
        type=type(value).__name__,
        start_index=start,
        stop_index=stop,
    )


def _non_white_pixels(img):
    return [
        (x, y)
        for x in range(img.width)
        for y in range(img.height)
        if img.getpixel((x, y)) != (255, 255, 255)
    ]


class TestParseFunctionCalls:
    def test_positional_args_are_named_from_signature(self):
        code = 'fill("a12", "hello")'
        args = overlay_utils.parse_function_calls(code)
        assert [(a.function_name, a.name, a.value, a.type) for a in args] == [
            ("fill", "bid", "a12", "str"),
            ("fill", "value", "hello", "str"),
        ]
        assert code[args[0].start_index : args[0].stop_index] == '"a12"'
        assert code[args[1].start_index : args[1].stop_index] == '"hello"'

    def test_keyword_arg_span_starts_at_keyword_name(self):
        code = 'click("b1", button="right")'
        args = overlay_utils.parse_function_calls(code)
        kw = [a for a in args if a.name == "button"][0]
        assert kw.value == "right"
        assert code[kw.start_index : kw.stop_index] == 'button="right"'

    def test_unknown_functions_and_attribute_calls_are_ignored(self):
        code = 'foo(1, 2)\nobj.click("a1")\nprint("x")'
        assert overlay_utils.parse_function_calls(code) == []

    def test_extra_positional_args_beyond_signature_are_dropped(self):
        n_params = len(inspect.signature(overlay_utils.BGYM_FUNCTION_MAP["focus"]).parameters)
        code = "focus(" + ", ".join(f"'a{i}'" for i in range(n_params + 3)) + ")"
        args = overlay_utils.parse_function_calls(code)
        assert len(args) == n_params
        assert args[0].value == "a0"

    def test_syntax_error_returns_empty_list(self, capsys):
        assert overlay_utils.parse_function_calls("click('a1'") == []
        assert "Syntax error" in capsys.readouterr().out

    def test_empty_string(self):
        assert overlay_utils.parse_function_calls("") == []

    def test_multiline_indices_account_for_previous_lines(self):
        code = "noop()\n\nmouse_click(12.5, 30)\n"
        args = overlay_utils.parse_function_calls(code)
        assert [a.value for a in args] == [12.5, 30]
        assert [a.type for a in args] == ["float", "int"]
        assert code[args[0].start_index : args[0].stop_index] == "12.5"
        assert code[args[1].start_index : args[1].stop_index] == "30"

    def test_signature_failure_is_skipped_with_warning(self, monkeypatch, capsys):
        def boom(_):
            raise ValueError("no sig")

        monkeypatch.setattr(overlay_utils.inspect, "signature", boom)
        assert overlay_utils.parse_function_calls('click("a1")') == []
        assert "Could not process function click" in capsys.readouterr().out


class TestExtractValue:
    @pytest.mark.parametrize(
        "src, expected",
        [
            ("'abc'", "abc"),
            ("42", 42),
            ("3.5", 3.5),
            ("None", None),
            ("True", True),
            ("['a', 1, ['b']]", ["a", 1, ["b"]]),
            ("some_var", "some_var"),
            ("1 + 2", "1 + 2"),
            ("{'k': 1}", "{'k': 1}"),
            ("-5", "-5"),
        ],
    )
    def test_values(self, src, expected):
        node = ast.parse(src, mode="eval").body
        assert overlay_utils._extract_value(node) == expected


class TestGetNodeIndices:
    def test_positional_node(self):
        src = "a\nfoo(bar)"
        call = ast.parse(src).body[1].value
        assert overlay_utils._get_node_indices(src, call.args[0]) == (6, 9)

    def test_keyword_node_uses_keyword_start(self):
        src = "foo(x=12)"
        kw = ast.parse(src).body[0].value.keywords[0]
        start, stop = overlay_utils._get_node_indices(src, kw.value, kw)
        assert src[start:stop] == "x=12"

    def test_fallback_when_no_end_positions(self):
        src = "foo(abc)"
        node = SimpleNamespace(lineno=1, col_offset=4)
        # get_source_segment returns None without end positions -> one-char fallback
        assert overlay_utils._get_node_indices(src, node) == (4, 5)

    def test_fallback_uses_source_segment_length(self, monkeypatch):
        src = "foo(abc)"
        node = SimpleNamespace(lineno=1, col_offset=4)
        monkeypatch.setattr(overlay_utils.ast, "get_source_segment", lambda s, n: "abc")
        assert overlay_utils._get_node_indices(src, node) == (4, 7)


class TestFindBidsAndXYPairs:
    def test_bids_kept_and_xy_merged(self):
        args = [
            _arg("bid", "a1", 0, 4),
            _arg("value", "text"),
            _arg("x", 10, 10, 12, fn="mouse_click"),
            _arg("y", "20", 14, 16, fn="mouse_click"),
        ]
        out = overlay_utils.find_bids_and_xy_pairs(args)
        assert [a.name for a in out] == ["bid", "xy"]
        merged = out[1]
        assert merged.value == (10.0, 20.0)
        assert merged.type == "tuple"
        assert (merged.start_index, merged.stop_index) == (10, 16)
        assert merged.function_name == "mouse_click"

    def test_prefixed_pairs(self):
        args = [
            _arg("from_x", 1, fn="d"),
            _arg("from_y", 2, fn="d"),
            _arg("to_x", 3, fn="d"),
            _arg("to_y", 4, fn="d"),
        ]
        out = overlay_utils.find_bids_and_xy_pairs(args)
        assert [(a.name, a.value) for a in out] == [("from_xy", (1.0, 2.0)), ("to_xy", (3.0, 4.0))]

    def test_mismatched_prefix_not_merged(self):
        args = [_arg("from_x", 1), _arg("to_y", 2)]
        assert overlay_utils.find_bids_and_xy_pairs(args) == []

    def test_different_function_not_merged(self):
        args = [_arg("x", 1, fn="a"), _arg("y", 2, fn="b")]
        assert overlay_utils.find_bids_and_xy_pairs(args) == []

    def test_non_numeric_values_kept_as_is(self):
        args = [_arg("x", "left"), _arg("y", None)]
        (merged,) = overlay_utils.find_bids_and_xy_pairs(args)
        assert merged.value == ("left", None)

    def test_trailing_x_without_y(self):
        assert overlay_utils.find_bids_and_xy_pairs([_arg("x", 1)]) == []

    def test_from_bid_and_to_bid(self):
        code = 'drag_and_drop("a1", "b2")'
        out = overlay_utils.find_bids_and_xy_pairs(overlay_utils.parse_function_calls(code))
        assert [(a.name, a.value) for a in out] == [("from_bid", "a1"), ("to_bid", "b2")]


class TestDrawing:
    def test_overlay_cross_draws_centered_plus(self):
        img = Image.new("RGB", (20, 20), "white")
        out = overlay_utils.overlay_cross(img, (10, 10), color=(255, 0, 0), length=7, width=1)
        assert out is img
        assert img.getpixel((10, 10)) == (255, 0, 0)
        assert img.getpixel((7, 10)) == (255, 0, 0)
        assert img.getpixel((13, 10)) == (255, 0, 0)
        assert img.getpixel((10, 7)) == (255, 0, 0)
        assert img.getpixel((10, 13)) == (255, 0, 0)
        assert img.getpixel((7, 7)) == (255, 255, 255)
        assert img.getpixel((15, 10)) == (255, 255, 255)

    def test_overlay_rectangle_solid(self):
        img = Image.new("RGB", (30, 30), "white")
        out = overlay_utils.overlay_rectangle(img, (5, 5, 10, 10), color=(0, 0, 255), dashed=False)
        assert out is img
        for p in [(5, 5), (15, 5), (5, 15), (15, 15), (10, 5), (5, 10)]:
            assert img.getpixel(p) == (0, 0, 255)
        assert img.getpixel((10, 10)) == (255, 255, 255)

    def test_overlay_rectangle_dashed_has_gaps(self):
        img = Image.new("RGB", (40, 40), "white")
        overlay_utils.overlay_rectangle(img, (2, 2, 30, 30), color=(0, 0, 255), dashed=True)
        top_edge = [img.getpixel((x, 2)) for x in range(2, 33)]
        assert (0, 0, 255) in top_edge
        assert (255, 255, 255) in top_edge
        assert img.getpixel((15, 15)) == (255, 255, 255)

    def test_linedashed_zero_length_draws_nothing(self):
        img = Image.new("RGB", (10, 10), "white")
        draw = overlay_utils.ImageDraw.Draw(img)
        overlay_utils.linedashed(draw, 5, 5, 5, 5, (0, 0, 0), 1)
        assert _non_white_pixels(img) == []

    def test_linedashed_pattern(self):
        img = Image.new("RGB", (40, 3), "white")
        draw = overlay_utils.ImageDraw.Draw(img)
        overlay_utils.linedashed(draw, 0, 1, 30, 1, (0, 0, 0), 1, dash_length=4, nodash_length=8)
        row = [img.getpixel((x, 1)) == (0, 0, 0) for x in range(40)]
        assert row[0:5] == [True] * 5
        assert row[5:12] == [False] * 7
        assert row[12] and row[24]
        assert not any(row[31:])

    def test_linedashed_negative_direction(self):
        img = Image.new("RGB", (3, 20), "white")
        draw = overlay_utils.ImageDraw.Draw(img)
        overlay_utils.linedashed(draw, 1, 18, 1, 0, (0, 0, 0), 1)
        assert img.getpixel((1, 18)) == (0, 0, 0)


class TestCreateColoredHtml:
    def test_spans_sorted_by_start_and_hex_colors(self):
        s = 'click("a1") then fill("b2", "x")'
        a1 = _arg("bid", "a1", s.index('"a1"'), s.index('"a1"') + 4)
        b2 = _arg("bid", "b2", s.index('"b2"'), s.index('"b2"') + 4)
        html = overlay_utils.create_colored_html(s, [b2, a1], [(0, 0, 1.0), (1.0, 0, 0)])
        assert html == (
            'click(<span style="color: #ff0000; font-weight: bold;">"a1"</span>) then fill('
            '<span style="color: #0000ff; font-weight: bold;">"b2"</span>, "x")'
        )

    def test_no_args_returns_original(self):
        assert overlay_utils.create_colored_html("noop()", [], []) == "noop()"

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: create_colored_html inserts raw action text into HTML without escaping",
    )
    def test_action_text_is_html_escaped(self):
        s = 'fill("a1", "<script>alert(1)</script>")'
        a1 = _arg("bid", "a1", 5, 9)
        html = overlay_utils.create_colored_html(s, [a1], [(1.0, 0, 0)])
        assert "<script>" not in html


class TestAnnotateAction:
    def test_bid_and_click_overlays_and_html(self):
        img = Image.new("RGB", (200, 200), "white")
        action = 'click("btn")\nmouse_click(150, 150)'
        props = {"btn": {"bbox": (10, 10, 50, 30)}}
        html = overlay_utils.annotate_action(img, action, props, colormap="tab10")

        assert html.count("<span") == 2
        assert '"btn"</span>' in html
        assert "150, 150</span>" in html
        # tab10 colors 0 and 1
        assert "#1f77b4" in html and "#ff7f0e" in html
        # rectangle drawn around (10,10)-(60,40); cross at (150,150)
        assert img.getpixel((10, 10)) != (255, 255, 255)
        assert img.getpixel((150, 150)) != (255, 255, 255)
        assert img.getpixel((100, 100)) == (255, 255, 255)

    def test_unknown_bid_and_empty_bbox_not_drawn(self):
        img = Image.new("RGB", (50, 50), "white")
        html = overlay_utils.annotate_action(
            img, 'click("missing")\nhover("nobox")', {"nobox": {"bbox": None}}
        )
        assert html.count("<span") == 2
        assert _non_white_pixels(img) == []

    def test_no_overlayable_args(self):
        img = Image.new("RGB", (10, 10), "white")
        assert overlay_utils.annotate_action(img, "noop()", {}) == "noop()"
        assert _non_white_pixels(img) == []
