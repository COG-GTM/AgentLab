# AgentLab

AgentLab is a Python framework for developing, evaluating, and analyzing AI agents that interact with web environments and desktop applications through standardized benchmarks.

<div align="center">

[![pypi](https://badge.fury.io/py/agentlab.svg)](https://pypi.org/project/agentlab/)
[![PyPI - License](https://img.shields.io/pypi/l/agentlab?style=flat-square)](http://www.apache.org/licenses/LICENSE-2.0)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/agentlab?style=flat-square)](https://pypistats.org/packages/agentlab)
[![GitHub star chart](https://img.shields.io/github/stars/ServiceNow/AgentLab?style=flat-square)](https://star-history.com/#ServiceNow/AgentLab)
[![Code Format](https://github.com/ServiceNow/AgentLab/actions/workflows/code_format.yml/badge.svg)](https://github.com/ServiceNow/AgentLab/actions/workflows/code_format.yml)
[![Tests](https://github.com/ServiceNow/AgentLab/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/ServiceNow/AgentLab/actions/workflows/unit_tests.yml)

<img src="https://github.com/user-attachments/assets/47a7c425-9763-46e5-be54-adac363be850" alt="agentlab-diagram" width="700"/>

[Demo solving tasks](https://github.com/ServiceNow/BrowserGym/assets/26232819/e0bfc788-cc8e-44f1-b8c3-0d1114108b85)

</div>

> [!WARNING]
> AgentLab is meant to provide an open, easy-to-use and extensible framework to accelerate the field of web agent research. It is not meant to be a consumer product. Use with caution!

## Table of Contents

- [Key Features](#key-features)
- [Supported Benchmarks](#supported-benchmarks)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [UI Assistant](#ui-assistant)
  - [Launch Experiments](#launch-experiments)
  - [Analyze Results](#analyze-results)
- [Leaderboard](#leaderboard)
- [Implement a New Agent](#implement-a-new-agent)
- [Reproducibility](#reproducibility)
- [Environment Variables](#environment-variables)
- [Further Documentation](#further-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citing This Work](#citing-this-work)

## Key Features

AgentLab provides a comprehensive toolkit for web agent research and development:

**Multi-Benchmark Support**: Evaluate agents across standardized benchmarks including WebArena, WorkArena, VisualWebArena, MiniWoB, OSWorld, AssistantBench, and more through the [BrowserGym](https://github.com/ServiceNow/BrowserGym) ecosystem.

**LLM Integration**: Unified API supporting multiple providers including OpenAI, Azure OpenAI, OpenRouter, Anthropic, and self-hosted models via TGI or LiteLLM.

**Experiment Management**: Run large-scale parallel experiments using [Ray](https://www.ray.io/) with automatic job scheduling, timeout handling, and result persistence.

**Analysis Tools**: Visualize and analyze agent behavior with AgentXray, a Gradio-based interface for exploring experiment traces, screenshots, and step-by-step actions.

**Reproducibility**: Built-in features for tracking software versions, benchmark states, and experiment configurations to ensure reproducible results.

## Supported Benchmarks

| Benchmark | Setup Link | Task Templates | Seed Diversity | Max Steps | Multi-tab | Hosted Method |
|-----------|------------|----------------|----------------|-----------|-----------|---------------|
| [WebArena](https://webarena.dev/) | [setup](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/webarena/README.md) | 812 | None | 30 | yes | self hosted (docker) |
| [WorkArena L1](https://github.com/ServiceNow/WorkArena) | [setup](https://github.com/ServiceNow/WorkArena?tab=readme-ov-file#getting-started) | 33 | High | 30 | no | demo instance |
| [WorkArena L2](https://github.com/ServiceNow/WorkArena) | [setup](https://github.com/ServiceNow/WorkArena?tab=readme-ov-file#getting-started) | 341 | High | 50 | no | demo instance |
| [WorkArena L3](https://github.com/ServiceNow/WorkArena) | [setup](https://github.com/ServiceNow/WorkArena?tab=readme-ov-file#getting-started) | 341 | High | 50 | no | demo instance |
| [WebLinx](https://mcgill-nlp.github.io/weblinx/) | - | 31586 | None | 1 | no | self hosted (dataset) |
| [VisualWebArena](https://github.com/web-arena-x/visualwebarena) | [setup](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/visualwebarena/README.md) | 910 | None | 30 | yes | self hosted (docker) |
| [AssistantBench](https://assistantbench.github.io/) | [setup](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/assistantbench/README.md) | 214 | None | 30 | yes | live web |
| [MiniWoB](https://miniwob.farama.org/index.html) | [setup](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/miniwob/README.md) | 125 | Medium | 10 | no | self hosted (static files) |
| [OSWorld](https://os-world.github.io/) | [setup](https://github.com/ServiceNow/AgentLab/blob/main/src/agentlab/benchmarks/osworld.md) | 369 | None | - | - | self hosted |

## Requirements

- Python 3.11 or 3.12
- [Playwright](https://playwright.dev/) for browser automation
- API keys for your chosen LLM provider (OpenAI, Azure, OpenRouter, or Anthropic)
- Benchmark-specific setup (see [Supported Benchmarks](#supported-benchmarks))

## Quick Start

### Installation

Install AgentLab from PyPI:

```bash
pip install agentlab
```

Install Playwright browser dependencies:

```bash
playwright install chromium
```

### Configuration

Set up your environment variables:

```bash
# Required: Directory for experiment results (defaults to ~/agentlab_results)
export AGENTLAB_EXP_ROOT=/path/to/your/experiments

# Required: API key for your LLM provider
export OPENAI_API_KEY=your-openai-api-key
```

<details>
<summary>OpenRouter API Setup</summary>

```bash
export OPENROUTER_API_KEY=your-openrouter-api-key
```
</details>

<details>
<summary>Azure OpenAI Setup</summary>

```bash
export AZURE_OPENAI_API_KEY=your-azure-api-key
export AZURE_OPENAI_ENDPOINT=your-azure-endpoint
export OPENAI_API_VERSION=2024-02-15-preview  # optional
```
</details>

<details>
<summary>Anthropic Setup</summary>

```bash
export ANTHROPIC_API_KEY=your-anthropic-api-key
```
</details>

### Run Your First Agent

Launch an interactive agent session:

```bash
agentlab-assistant --start_url https://www.google.com
```

## Project Structure

```
src/agentlab/
├── agents/                 # Agent implementations
│   ├── generic_agent/      # Configurable baseline agent with dynamic prompting
│   ├── most_basic_agent/   # Minimal agent example for reference
│   ├── tool_use_agent/     # Agent specialized for function calling
│   ├── tapeagent/          # Agent with execution recording
│   └── visual_agent/       # Vision-enabled agent
├── analyze/                # Result analysis and visualization
│   ├── agent_xray.py       # Gradio-based experiment explorer
│   ├── inspect_results.py  # Result loading and reporting utilities
│   └── episode_to_html.py  # HTML visualization generation
├── benchmarks/             # Benchmark environment wrappers
│   ├── osworld.py          # OSWorld desktop automation
│   ├── gaia.py             # GAIA multi-tool tasks
│   └── abstract_env.py     # Base benchmark interface
├── experiments/            # Experiment orchestration
│   ├── study.py            # Study management and execution
│   ├── loop.py             # Agent-environment interaction loop
│   └── graph_execution_ray.py  # Parallel execution with Ray
└── llm/                    # LLM API integration
    ├── chat_api.py         # Multi-provider chat interface
    ├── llm_configs.py      # Pre-configured model settings
    └── tracking.py         # Token usage and cost tracking
```

## Usage

### UI Assistant

Use an interactive agent to browse the web on your behalf:

```bash
agentlab-assistant --start_url https://www.google.com
```

Use a custom agent configuration:

```bash
agentlab-assistant --agent_config="module.path.to.your.AgentArgs"
```

### Launch Experiments

Run a study across a benchmark:

```python
from agentlab.agents.generic_agent import AGENT_4o_MINI
from agentlab.experiments.study import make_study

study = make_study(
    benchmark="miniwob",  # or "webarena", "workarena_l1", etc.
    agent_args=[AGENT_4o_MINI],
    comment="My first study",
)

study.run(n_jobs=5)
```

Relaunch incomplete or errored tasks:

```python
from agentlab.experiments.study import Study

study = Study.load("/path/to/your/study/dir")
study.find_incomplete(include_errors=True)
study.run()
```

See [main.py](main.py) for additional experiment configuration options.

#### Job Timeouts

The complexity of web environments can sometimes cause jobs to hang. The Ray parallel backend includes automatic timeout handling to terminate stuck jobs and maintain experiment progress.

#### Debugging

For debugging, run experiments with `n_jobs=1` and use your IDE debug mode to set breakpoints and step through execution.

#### Parallel Execution Notes

Running one agent on one task corresponds to a single job. Ablation studies across hundreds of tasks can generate thousands of jobs, making efficient parallel execution critical. Agents typically wait for LLM responses or web server updates, allowing 10-50 parallel jobs on a single machine depending on available RAM.

**Note for WebArena/VisualWebArena**: These benchmarks have task dependencies to prevent state corruption between tasks. The Ray backend accounts for these dependencies while still enabling parallelism. Before evaluating an agent, the instance is automatically reset (approximately 5 minutes).

### Analyze Results

#### Loading Results

```python
from agentlab.analyze import inspect_results
import browsergym as bgym

# Load study results into a DataFrame
result_df = inspect_results.load_result_df("/path/to/your/study")

# Load detailed results for a specific experiment
exp_result = bgym.ExpResult(result_df["exp_dir"][0])
step_0_screenshot = exp_result.screenshots[0]
step_0_action = exp_result.steps_info[0].action
```

See [inspect_results.ipynb](src/agentlab/analyze/inspect_results.ipynb) for detailed analysis examples.

#### AgentXray

Launch the visual experiment explorer:

```bash
agentlab-xray
```

AgentXray provides a Gradio interface to explore experiments in your `AGENTLAB_EXP_ROOT` directory. Select an experiment, agent, task, and seed to view the step-by-step trace with screenshots and actions.

https://github.com/user-attachments/assets/06c4dac0-b78f-45b7-9405-003da4af6b37

## Leaderboard

View the official unified [leaderboard](https://huggingface.co/spaces/ServiceNow/browsergym-leaderboard) for benchmark results across all supported environments.

## Implement a New Agent

Create a custom agent by following the example in [most_basic_agent.py](src/agentlab/agents/most_basic_agent/most_basic_agent.py). For full integration with AgentLab tools, implement the [AgentArgs](src/agentlab/agents/agent_args.py) API and extend `bgym.AbstractAgentArgs`.

If you develop an agent that could benefit the community, consider contributing it to the `agentlab/agents/` directory.

## Reproducibility

Several factors can affect reproducibility when evaluating agents on dynamic benchmarks:

**Software Versions**: Different versions of Playwright or other dependencies may influence benchmark or agent behavior.

**API-based LLMs**: Even for a fixed version, LLMs may be silently updated by providers.

**Live Websites**: WorkArena uses a mostly-fixed demo instance, while AssistantBench and GAIA rely on the open web where content varies by region and time.

**Stochastic Agents**: Setting LLM temperature to 0 reduces most stochasticity.

**Non-deterministic Tasks**: For a fixed seed, changes should be minimal.

### Reproducibility Features

- **Study Metadata**: Each `Study` contains version information including benchmark version, package versions, and commit hashes.
- **Reproducibility Journal**: Automatic upload of results to [reproducibility_journal.csv](reproducibility_journal.csv) for reference tracking. Requires installation via `pip install -e .` from a cloned repository.
- **Leaderboard Reproduction**: The leaderboard includes a column for reproduced results, encouraging validation of published agent performance.
- **ReproducibilityAgent**: [Run this agent](src/agentlab/agents/generic_agent/reproducibility_agent.py) on an existing study to replay actions and compare execution traces with visual diffs in AgentXray.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | - |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | - |
| `OPENAI_API_VERSION` | Azure API version | - |
| `OPENROUTER_API_KEY` | OpenRouter API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `AGENTLAB_EXP_ROOT` | Experiment results directory | `~/agentlab_results` |
| `AGENTXRAY_SHARE_GRADIO` | Open public Gradio tunnel | `false` |

### Optional: Faster Model Downloads

For faster HuggingFace model downloads:

```bash
pip install hf-transfer torch
export HF_HUB_ENABLE_HF_TRANSFER=1
```

## Further Documentation

- [BrowserGym Documentation](https://github.com/ServiceNow/BrowserGym) - Core browser automation framework
- [BrowserGym Ecosystem Paper](https://arxiv.org/abs/2412.05467) - Detailed technical description
- [OSWorld Setup Guide](src/agentlab/benchmarks/osworld.md) - Desktop automation benchmark configuration
- [LLM Integration Guide](src/agentlab/llm/README.md) - Detailed LLM provider configuration

## Contributing

Contributions are welcome! AgentLab is an open framework designed to accelerate web agent research.

To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes following the existing code style
4. Run tests with `make test`
5. Submit a pull request

For development setup:

```bash
git clone https://github.com/ServiceNow/AgentLab.git
cd AgentLab
pip install -e ".[dev]"
playwright install chromium --with-deps
```

Run code formatting:

```bash
black src/ --check --diff
```

## License

AgentLab is licensed under the [Apache License 2.0](LICENSE).

## Citing This Work

If you use AgentLab in your research, please cite:

```bibtex
@article{chezelles2025browsergym,
    title={The BrowserGym Ecosystem for Web Agent Research},
    author={Thibault Le Sellier de Chezelles and Maxime Gasse and Alexandre Lacoste and Massimo Caccia and Alexandre Drouin and L{'''e}o Boisvert and Megh Thakkar and Tom Marty and Rim Assouel and Sahar Omidi Shayegan and Lawrence Keunho Jang and Xing Han L{'`''u} and Ori Yoran and Dehan Kong and Frank F. Xu and Siva Reddy and Graham Neubig and Quentin Cappart and Russ Salakhutdinov and Nicolas Chapados},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2025},
    url={https://openreview.net/forum?id=5298fKGmv3},
    note={Expert Certification}
}

@inproceedings{workarena2024,
    title={{W}ork{A}rena: How Capable are Web Agents at Solving Common Knowledge Work Tasks?},
    author={Drouin, Alexandre and Gasse, Maxime and Caccia, Massimo and Laradji, Issam H. and Del Verme, Manuel and Marty, Tom and Vazquez, David and Chapados, Nicolas and Lacoste, Alexandre},
    booktitle={Proceedings of the 41st International Conference on Machine Learning},
    pages={11642--11662},
    year={2024},
    editor={Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
    volume={235},
    series={Proceedings of Machine Learning Research},
    month={21--27 Jul},
    publisher={PMLR},
    url={https://proceedings.mlr.press/v235/drouin24a.html},
}
```

Example usage in papers:

```tex
We use the AgentLab framework to run and manage our experiments \cite{workarena2024,chezelles2025browsergym}.
```

---

_Originally written and maintained by contributors and [Devin](https://app.devin.ai), with updates from the core team._
