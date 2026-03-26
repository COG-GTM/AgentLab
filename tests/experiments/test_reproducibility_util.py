import json
import tempfile
import time
from pathlib import Path

import bgym
import pytest
from bgym import DEFAULT_BENCHMARKS

from agentlab.agents.generic_agent import AGENT_4o_MINI
from agentlab.analyze import inspect_results
from agentlab.experiments import reproducibility_util


@pytest.mark.parametrize(
    "benchmark_name",
    ["miniwob", "workarena_l1", "webarena", "visualwebarena"],
)
def test_get_reproducibility_info(benchmark_name):

    benchmark = DEFAULT_BENCHMARKS[benchmark_name]()

    info = reproducibility_util.get_reproducibility_info(
        "test_agent", benchmark, "test_id", ignore_changes=True
    )

    print("reproducibility info:")
    print(json.dumps(info, indent=4))

    # assert keys in info
    assert "git_user" in info
    assert "benchmark" in info
    assert "benchmark_version" in info
    assert "agentlab_version" in info
    assert "agentlab_git_hash" in info
    assert "agentlab__local_modifications" in info
    assert "browsergym_version" in info
    assert "browsergym_git_hash" in info
    assert "browsergym__local_modifications" in info


def test_get_reproducibility_info_and_assert_compatible():
    """Test get_reproducibility_info + assert_compatible (replaces old save_reproducibility_info test).

    The original test used save_reproducibility_info/load_reproducibility_info which no longer
    exist. This test exercises the current API: get_reproducibility_info and assert_compatible.
    """
    benchmark = DEFAULT_BENCHMARKS["miniwob"]()

    info1 = reproducibility_util.get_reproducibility_info(
        "GenericAgent", benchmark, "test_id", ignore_changes=True
    )
    time.sleep(1)  # make sure the date changes by at least 1s

    info2 = reproducibility_util.get_reproducibility_info(
        "GenericAgent", benchmark, "test_id", ignore_changes=True
    )

    # Compatible infos should not raise
    reproducibility_util.assert_compatible(info1, info2)

    # Different agent name should raise
    info3 = reproducibility_util.get_reproducibility_info(
        "GenericAgent_alt", benchmark, "test_id", ignore_changes=True
    )
    with pytest.raises(ValueError):
        reproducibility_util.assert_compatible(info1, info3)


if __name__ == "__main__":
    # test_set_temp()
    test_get_reproducibility_info("miniwob")
    # test_save_reproducibility_info()
