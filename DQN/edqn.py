"""Compatibility shim for historical route-eDQN checkpoints."""

from css.agents.routing.edqn import (  # noqa: F401
    EDQNShapeConfig,
    route_eDQN,
)

EDQNAgentRefactored = route_eDQN

__all__ = ["EDQNAgentRefactored", "EDQNShapeConfig", "route_eDQN"]
