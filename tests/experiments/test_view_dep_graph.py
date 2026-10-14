import importlib
import sys
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import bgym  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pytest  # noqa: E402

MODULE = "agentlab.experiments.view_dep_graph"

FAKE_DEPS = {
    "visualwebarena.1": [],
    "visualwebarena.2": ["visualwebarena.1"],
    "visualwebarena.3": ["visualwebarena.2"],
    "visualwebarena.7": ["visualwebarena.3"],
    "visualwebarena.9": [],
    "visualwebarena.10": ["visualwebarena.9"],
}


class FakeBenchmark:
    instances = 0

    def __init__(self):
        FakeBenchmark.instances += 1

    def dependency_graph_over_tasks(self):
        return dict(FAKE_DEPS)


@pytest.fixture(scope="module")
def import_result():
    """Import the script-like module with a fake benchmark so its top-level code runs offline."""
    FakeBenchmark.instances = 0
    with pytest.MonkeyPatch.context() as mp:
        mp.delitem(sys.modules, MODULE, raising=False)
        mp.setitem(bgym.DEFAULT_BENCHMARKS, "visualwebarena", FakeBenchmark)
        show = mock.Mock()
        mp.setattr(plt, "show", show)
        module = importlib.import_module(MODULE)
        n_figs = len(plt.get_fignums())
        instances = FakeBenchmark.instances
    plt.close("all")
    return module, show, n_figs, instances


@pytest.fixture
def vdg(import_result):
    yield import_result[0]
    plt.close("all")


def test_module_script_runs_with_benchmark_graph(import_result):
    _, show, n_figs, instances = import_result
    assert instances == 1
    show.assert_called_once()
    assert n_figs == 1


def test_clean_dict_strips_benchmark_prefix(vdg):
    assert vdg.clean_dict({"a.b.1": ["a.b.2", "x.3"], "4": []}) == {"1": ["2", "3"], "4": []}


def test_dict_to_networkx_edges_point_from_dependency_to_task(vdg):
    G = vdg.dict_to_networkx({"2": ["1"], "3": ["1", "2"], "lonely": []})
    assert isinstance(G, nx.DiGraph)
    assert set(G.edges) == {("1", "2"), ("1", "3"), ("2", "3")}
    # nodes without any dependency relation are never added
    assert "lonely" not in G


def test_dict_to_networkx_many_nodes(vdg):
    deps = {str(i): [str(i - 1)] for i in range(1, 25)}
    G = vdg.dict_to_networkx(deps)
    assert G.number_of_nodes() == 25
    assert G.number_of_edges() == 24


@pytest.mark.parametrize(
    "deps, expected",
    [
        (
            {"1": [], "2": ["1"], "3": ["2"], "5": []},
            {"1-3": [], "5": []},
        ),
        (
            {"1": [], "2": ["1"], "3": ["2"], "10": ["3"]},
            {"1-3": [], "10": ["1-3"]},
        ),
        (
            # node 1 has two successors -> no chain
            {"1": [], "2": ["1"], "3": ["1"]},
            {"1": [], "2": ["1"], "3": ["1"]},
        ),
        (
            {"1": [], "2": ["1"], "4": [], "5": ["4"], "6": ["5"]},
            {"1-2": [], "4-6": []},
        ),
        (
            # chain starting after an external dependency keeps it
            {"1": [], "5": ["1"], "6": ["5"], "7": ["6"]},
            {"5-7": ["1"], "1": []},
        ),
    ],
)
def test_compress_sequential_chains(vdg, deps, expected):
    assert vdg.compress_sequential_chains(deps) == expected


def test_compress_chains_linear_graph_collapses_fully(vdg):
    G = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")])
    compressed = vdg.compress_chains(G)
    assert list(compressed.nodes) == ["a,b,c,d,e"]
    assert G.number_of_nodes() == 5  # original untouched


def test_compress_chains_keeps_external_edges(vdg):
    G = nx.DiGraph([("root1", "x"), ("root2", "x"), ("x", "a"), ("a", "b"), ("b", "c")])
    compressed = vdg.compress_chains(G)
    assert "x,a,b" not in compressed  # x has two predecessors
    assert set(compressed.nodes) == {"root1", "root2", "x", "a,b,c"}
    assert set(compressed.edges) == {("root1", "x"), ("root2", "x"), ("x", "a,b,c")}


def test_compress_chains_branching_graph_unchanged(vdg):
    G = nx.DiGraph([("x", "y"), ("x", "z")])
    compressed = vdg.compress_chains(G)
    assert compressed is not G
    assert set(compressed.edges) == set(G.edges)


def test_plot_graph_sizes_nodes_by_range_and_sets_title(vdg):
    G = nx.DiGraph([("1-3", "5")])
    fig, ax = plt.subplots()
    with mock.patch.object(vdg.nx, "draw") as draw:
        vdg.plot_graph(G, ax, title="My Title", node_size=40)
    sizes = dict(zip(G.nodes, draw.call_args.kwargs["node_size"]))
    assert sizes == {"1-3": 12.0, "5": 4.0}
    assert draw.call_args.kwargs["ax"] is ax
    assert ax.get_title() == "My Title"
    assert not ax.axison


def test_plot_graph_real_draw_without_title(vdg):
    G = nx.DiGraph([("1", "2"), ("2", "3")])
    fig, ax = plt.subplots()
    vdg.plot_graph(G, ax)
    assert ax.get_title() == ""
    assert len(ax.collections) > 0


def test_plot_components_grid_empty(vdg, capsys):
    assert vdg.plot_components_grid([]) is None
    assert "No components found" in capsys.readouterr().out


def _components(n):
    return [nx.DiGraph([(f"{i}a", f"{i}b")]) for i in range(n)]


@pytest.mark.parametrize(
    "n_components, max_cols, n_axes",
    [(1, 4, 1), (3, 4, 3), (2, 1, 2), (3, 2, 3), (4, 2, 4)],
)
def test_plot_components_grid_layouts(vdg, n_components, max_cols, n_axes):
    with mock.patch.object(vdg.nx, "draw"):
        fig = vdg.plot_components_grid(_components(n_components), max_cols=max_cols)
    assert len(fig.axes) == n_axes
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == [f"Component {i + 1} (2 nodes)" for i in range(n_components)]
    assert fig._suptitle.get_text() == "Dependency Graph Components"
