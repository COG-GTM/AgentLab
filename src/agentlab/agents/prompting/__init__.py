"""Prompting subpackage for AgentLab agents.

This package provides modular prompt construction components split into:
- flags: Configuration flags (Flags, ObsFlags, ActionFlags)
- base: Base prompt element classes (PromptElement, Shrinkable, Trunkater, fit_tokens)
- elements: Concrete prompt element implementations
"""

from .base import PromptElement, Shrinkable, Trunkater, fit_tokens
from .elements import (
    HTML,
    AXTree,
    ActionPrompt,
    BeCautious,
    ChatInstructions,
    Error,
    FocusedElement,
    GoalInstructions,
    Hints,
    History,
    HistoryStep,
    MacNote,
    Observation,
    SystemPrompt,
    Tabs,
    Think,
    make_obs_preprocessor,
)
from .flags import ActionFlags, Flags, ObsFlags

__all__ = [
    "Flags",
    "ObsFlags",
    "ActionFlags",
    "PromptElement",
    "Shrinkable",
    "Trunkater",
    "fit_tokens",
    "HTML",
    "AXTree",
    "Error",
    "FocusedElement",
    "Tabs",
    "Observation",
    "MacNote",
    "BeCautious",
    "GoalInstructions",
    "ChatInstructions",
    "Hints",
    "SystemPrompt",
    "ActionPrompt",
    "Think",
    "HistoryStep",
    "History",
    "make_obs_preprocessor",
]
