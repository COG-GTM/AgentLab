import gzip
import json
import pickle

import numpy as np
import pytest

from agentlab.experiments.loop import DataclassJSONEncoder, ExpResult, StepInfo, StepTimestamps

GOAL_OBJECT = [{"type": "text", "text": "do the thing"}]


def _make_step_info(step=0):
    rng = np.random.default_rng(0)
    screenshot = rng.integers(0, 256, size=(4, 5, 3), dtype=np.uint8)
    screenshot_som = rng.integers(0, 256, size=(4, 5, 3), dtype=np.uint8)
    obs = {
        "screenshot": screenshot,
        "screenshot_som": screenshot_som,
        "goal_object": list(GOAL_OBJECT),
        "axtree_txt": "RootWebArea",
    }
    return StepInfo(
        step=step,
        obs=obs,
        reward=np.float32(0.5),
        raw_reward=np.int64(1),
        terminated=False,
        truncated=False,
        action="click('12')",
        agent_info={"think": "some thought"},
        stats={"n_token_axtree_txt": np.int64(3)},
        profiling=StepTimestamps(env_start=1.0, env_stop=2.0),
    )


def test_save_step_info_round_trip(tmp_path):
    step_info = _make_step_info()
    screenshot = step_info.obs["screenshot"]
    screenshot_som = step_info.obs["screenshot_som"]
    goal_object = step_info.obs["goal_object"]

    step_info.save_step_info(tmp_path, save_screenshot=True, save_som=True)

    assert (tmp_path / "screenshot_step_0.png").exists()
    assert (tmp_path / "screenshot_som_step_0.png").exists()
    assert (tmp_path / "goal_object.pkl.gz").exists()

    # obs is restored in memory, except for the offloaded goal object
    np.testing.assert_array_equal(step_info.obs["screenshot"], screenshot)
    np.testing.assert_array_equal(step_info.obs["screenshot_som"], screenshot_som)
    assert step_info.obs["goal_object"] is None

    # the pickled step contains neither the screenshots nor the goal object
    with gzip.open(tmp_path / "step_0.pkl.gz", "rb") as f:
        raw_step = pickle.load(f)
    assert "screenshot" not in raw_step.obs
    assert "screenshot_som" not in raw_step.obs
    assert raw_step.obs["goal_object"] is None

    loaded = ExpResult(tmp_path).get_step_info(0)
    np.testing.assert_array_equal(loaded.obs["screenshot"], screenshot)
    np.testing.assert_array_equal(loaded.obs["screenshot_som"], screenshot_som)
    assert loaded.obs["goal_object"] == goal_object
    assert loaded.action == step_info.action
    assert loaded.obs["axtree_txt"] == step_info.obs["axtree_txt"]


def test_save_step_info_goal_object_saved_once(tmp_path):
    first = _make_step_info(step=0)
    first.save_step_info(tmp_path)

    second = _make_step_info(step=1)
    second.obs["goal_object"] = [{"type": "text", "text": "a different goal"}]
    second.save_step_info(tmp_path)

    with gzip.open(tmp_path / "goal_object.pkl.gz", "rb") as f:
        assert pickle.load(f) == GOAL_OBJECT

    assert ExpResult(tmp_path).get_step_info(1).obs["goal_object"] == GOAL_OBJECT


def test_save_step_info_without_screenshots(tmp_path):
    step_info = _make_step_info()
    step_info.save_step_info(tmp_path, save_screenshot=False, save_som=False)

    assert not (tmp_path / "screenshot_step_0.png").exists()
    assert not (tmp_path / "screenshot_som_step_0.png").exists()

    loaded = ExpResult(tmp_path).get_step_info(0)
    assert "screenshot" not in loaded.obs
    assert "screenshot_som" not in loaded.obs


def test_save_step_info_json(tmp_path):
    step_info = _make_step_info()
    step_info.obs["numpy_array"] = np.arange(3, dtype=np.int64)
    step_info.save_step_info(tmp_path, save_json=True)

    with open(tmp_path / "steps_info.json", "r") as f:
        data = json.load(f)

    assert data["step"] == 0
    assert data["reward"] == pytest.approx(0.5)
    assert data["raw_reward"] == 1
    assert data["stats"]["n_token_axtree_txt"] == 3
    assert data["obs"]["numpy_array"] == [0, 1, 2]
    assert data["obs"]["goal_object"] is None


def test_dataclass_json_encoder_numpy():
    encoded = json.dumps(
        {
            "int": np.int64(3),
            "float": np.float32(1.5),
            "array": np.arange(2),
            "dataclass": StepTimestamps(env_start=1.0),
        },
        cls=DataclassJSONEncoder,
    )
    decoded = json.loads(encoded)

    assert decoded["int"] == 3
    assert decoded["float"] == pytest.approx(1.5)
    assert decoded["array"] == [0, 1]
    assert decoded["dataclass"]["env_start"] == 1.0
