from dataclasses import dataclass, field

import numpy as np
import pytest

from agentlab.experiments import args as args_mod
from agentlab.experiments.args import (
    TOGGLE,
    Choice,
    CrossProd,
    Distribution,
    Toggle,
    _apply_change,
    _change_value,
    _find_cprod_with_paths,
    _sample_single,
    _set_value,
    expand_cross_product,
    make_ablation_study,
    make_progression_study,
    sample_and_expand_cross_product,
    sample_args,
)


@dataclass
class Inner:
    name: str = "a"
    flag: bool = False
    value: int = 0


@dataclass
class Outer:
    inner: Inner = field(default_factory=Inner)
    count: int = 1
    extra: dict = field(default_factory=dict)


class PlainObj:
    def __init__(self):
        self.inner = Inner()
        self.flag = True
        self.label = "x"


def test_expand_top_level_cross_prod_returns_elements():
    assert expand_cross_product(CrossProd([1, 2, 3])) == [1, 2, 3]


def test_expand_without_cross_prod_returns_deep_copy():
    obj = Outer(extra={"k": [1]})
    result = expand_cross_product(obj)
    assert result == [obj]
    assert result[0] is not obj
    assert result[0].extra is not obj.extra


def test_expand_list_input_mixes_plain_and_cross_prod_objects():
    plain = Outer(count=7)
    crossed = Outer(count=CrossProd([1, 2]), inner=Inner(name=CrossProd(["x", "y"])))
    result = expand_cross_product([plain, crossed])
    assert len(result) == 1 + 4
    assert result[0].count == 7
    combos = sorted((o.count, o.inner.name) for o in result[1:])
    assert combos == [(1, "x"), (1, "y"), (2, "x"), (2, "y")]


def test_expand_does_not_mutate_original():
    obj = Outer(count=CrossProd([1, 2]))
    expand_cross_product(obj)
    assert isinstance(obj.count, CrossProd)


def test_expand_cross_prod_inside_dict_field_of_dataclass():
    obj = Outer(extra={"lr": CrossProd([0.1, 0.2])})
    result = expand_cross_product(obj)
    assert sorted(o.extra["lr"] for o in result) == [0.1, 0.2]


def test_find_cprod_with_paths_reports_nested_paths():
    cp1, cp2 = CrossProd([1]), CrossProd([2])
    obj = Outer(inner=Inner(name=cp1), extra={"a": {"b": cp2}})
    found = _find_cprod_with_paths(obj)
    assert (["inner", "name"], cp1) in found
    assert (["extra", "a", "b"], cp2) in found
    assert len(found) == 2


@pytest.mark.parametrize("obj", [1, "str", [CrossProd([1])], None])
def test_find_cprod_ignores_non_container_types(obj):
    # lists are not traversed, only dataclasses and dicts
    assert _find_cprod_with_paths(obj) == []


def test_set_value_traverses_dicts_and_objects():
    obj = Outer(extra={"nested": Inner()})
    _set_value(obj, ["extra", "nested", "value"], 42)
    _set_value(obj, ["inner", "name"], "z")
    _set_value(obj, ["extra", "new"], 1)
    assert obj.extra["nested"].value == 42
    assert obj.inner.name == "z"
    assert obj.extra["new"] == 1


def test_distribution_base_sample_returns_none():
    assert Distribution().sample() is None


def test_choice_respects_probabilities():
    choice = Choice(["a", "b", "c"], p=[0.0, 1.0, 0.0])
    assert {choice.sample() for _ in range(10)} == {"b"}


def test_choice_uses_given_rng():
    choice = Choice([1, 2, 3, 4, 5])
    s1 = [choice.sample(np.random.RandomState(0)) for _ in range(3)]
    s2 = [choice.sample(np.random.RandomState(0)) for _ in range(3)]
    assert s1 == s2


def test_sample_args_list_input_and_deep_copies():
    a = Outer(count=Choice([5], p=[1.0]))
    b = {"x": Choice(["only"])}
    result = sample_args([a, b], 2)
    assert len(result) == 4
    assert [r.count for r in result[:2]] == [5, 5]
    assert [r["x"] for r in result[2:]] == ["only", "only"]
    assert isinstance(a.count, Choice)
    assert isinstance(b["x"], Choice)
    assert result[0] is not result[1]


def test_sample_single_top_level_distribution():
    assert _sample_single(Choice([9])) == 9


def test_sample_single_leaves_plain_values_untouched():
    assert _sample_single(3) == 3


def test_sample_and_expand_cross_product():
    obj = Outer(count=Choice([3]), inner=Inner(name=CrossProd(["p", "q"])))
    result = sample_and_expand_cross_product(obj, n_samples=2)
    assert len(result) == 4
    assert all(r.count == 3 for r in result)
    assert sorted(r.inner.name for r in result) == ["p", "p", "q", "q"]


@pytest.mark.parametrize("path", ["inner.name", ".inner.name"])
def test_change_value_nested_path_with_optional_leading_dot(path):
    obj = Outer()
    _change_value(obj, path, "new")
    assert obj.inner.name == "new"


def test_change_value_unknown_dataclass_field_raises():
    with pytest.raises(ValueError, match="field missing not found"):
        _change_value(Outer(), "inner.missing", 1)


def test_change_value_on_plain_object_sets_any_attribute():
    obj = PlainObj()
    _change_value(obj, "brand_new", 5)
    assert obj.brand_new == 5


def test_change_value_toggle_flips_boolean_on_dataclass_and_object():
    obj = Outer()
    _change_value(obj, "inner.flag", TOGGLE)
    assert obj.inner.flag is True
    _change_value(obj, "inner.flag", TOGGLE)
    assert obj.inner.flag is False

    plain = PlainObj()
    _change_value(plain, "flag", Toggle())
    assert plain.flag is False


def test_change_value_toggle_non_boolean_raises():
    with pytest.raises(ValueError, match="not a boolean"):
        _change_value(Outer(), "inner.name", TOGGLE)


def test_apply_change_callable():
    obj = Outer()

    def change(params):
        params.count = 99

    assert _apply_change(obj, change) is obj
    assert obj.count == 99


def test_apply_change_list_of_changes_and_nested():
    obj = Outer()
    _apply_change(obj, [("count", 3), [("inner.name", "n"), ("inner.value", 4)]])
    assert (obj.count, obj.inner.name, obj.inner.value) == (3, "n", 4)


@pytest.mark.parametrize("bad", [42, "count", None])
def test_apply_change_unrecognized_raises(bad):
    with pytest.raises(ValueError, match="not recognized"):
        _apply_change(Outer(), bad)


def test_make_progression_study_accumulates_and_returns_list():
    start = Inner()
    result = make_progression_study(
        start, [("name", "b"), ("value", 2), ("flag", TOGGLE)], return_cross_prod=False
    )
    assert isinstance(result, list)
    assert [(r.name, r.value, r.flag) for r in result] == [
        ("a", 0, False),
        ("b", 0, False),
        ("b", 2, False),
        ("b", 2, True),
    ]
    assert result[0] is start
    assert start == Inner()


def test_make_ablation_study_resets_each_time_and_returns_list():
    start = Inner()
    result = make_ablation_study(
        start, [("name", "b"), ("value", 2), ("flag", TOGGLE)], return_cross_prod=False
    )
    assert [(r.name, r.value, r.flag) for r in result] == [
        ("a", 0, False),
        ("b", 0, False),
        ("a", 2, False),
        ("a", 0, True),
    ]
    assert start == Inner()


@pytest.mark.parametrize("fn", [make_progression_study, make_ablation_study])
def test_study_builders_return_cross_prod_by_default(fn):
    result = fn(Inner(), [("value", 1)])
    assert isinstance(result, CrossProd)
    assert len(result.elements) == 2


def test_toggle_singleton_is_toggle():
    assert isinstance(args_mod.TOGGLE, Toggle)
