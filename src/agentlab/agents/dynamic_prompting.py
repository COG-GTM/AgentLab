"""Dynamic prompting module for AgentLab agents.

This module re-exports all symbols from the prompting subpackage for backward compatibility.
New code should import directly from agentlab.agents.prompting instead.
"""

# Re-export everything from the prompting subpackage for backward compatibility
from agentlab.agents.prompting import (  # noqa: F401
    HTML,
    AXTree,
    ActionFlags,
    ActionPrompt,
    BeCautious,
    ChatInstructions,
    Error,
    Flags,
    FocusedElement,
    GoalInstructions,
    Hints,
    History,
    HistoryStep,
    MacNote,
    Observation,
    ObsFlags,
    PromptElement,
    Shrinkable,
    SystemPrompt,
    Tabs,
    Think,
    Trunkater,
    fit_tokens,
    make_obs_preprocessor,
)
