"""Gymnasium API. Install with ``uv sync --extra environment``."""
from gymnasium.envs.registration import register, registry
from .config import EnvironmentConfig
from .env import KuramotoEnv

ENV_ID = "KuramotoTemplate-v0"
if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="kuramoto_template.environment:KuramotoEnv")

__all__ = ["ENV_ID", "EnvironmentConfig", "KuramotoEnv"]
