"""Gymnasium adapter for serial phase-template mutation episodes.

The default evaluator is uncached and exact-data deduplication is disabled:
revisiting a state costs actual solver calls, avoiding hidden history-dependent
costs in the first RL task. Search algorithms keep their existing deduplication.
"""
from dataclasses import asdict
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import ResetNeeded
import numpy as np

from kuramoto_template.experiments.cache import CachedEvaluator
from kuramoto_template.search.evaluator import EvaluationConfig, TemplateEvaluator
from kuramoto_template.search.infrastructure import SearchSession
from kuramoto_template.search.random_search import twisted_seed
from kuramoto_template.template import PhaseTemplate
from .actions import action_space, action_masks, decode_action
from .config import EnvironmentConfig

STATUSES = ("stable", "near_feasible", "uncertain", "infeasible", "solver_error", "invalid")
GLOBAL_FEATURES = ("best_mu", "current_stable_mu", "margin_gamma", "cluster_fraction",
                   "remaining_sdp_calls", "max_sdp_calls", "step_fraction", "zero_cost_fraction")
DIAGNOSTIC_FEATURES = ("mu", "gamma", "min_restoring", "min_cluster_eigenvalue")


def build_observation(t, e, c, *, best_mu, remaining_sdp_calls, steps=0, zero_cost_steps=0):
    """Encode a stored state without invoking a solver; same contract as live env."""
    m, M = t.n_clusters, c.max_clusters
    nodes = np.zeros((M, 2), dtype=np.float32)
    nodes[:m, 0], nodes[:m, 1] = t.x, np.log(t.x)
    edges = np.zeros((M, M, 2), dtype=np.float32)
    diff = t.phi[:, None] - t.phi[None, :]
    edges[:m, :m, 0], edges[:m, :m, 1] = np.sin(diff), np.cos(diff)
    operations, node_mask, pairs = action_masks(t, c)
    diagnostics = np.zeros((2, 4), dtype=np.float32)
    diagnostic_mask = np.zeros((2, 4), dtype=np.int8)
    limit = np.finfo(np.float32).max
    for row, diagnostic in enumerate((e.density, e.margin)):
        if diagnostic is not None:
            for col, name in enumerate(DIAGNOSTIC_FEATURES):
                value = getattr(diagnostic, name)
                if value is not None and np.isfinite(value) and abs(value) <= limit:
                    diagnostics[row, col], diagnostic_mask[row, col] = value, 1
    status = np.array([e.status == name for name in STATUSES], dtype=np.int8)
    gamma = e.margin_gamma if e.margin_gamma is not None and np.isfinite(e.margin_gamma) and abs(e.margin_gamma) <= limit else 0.0
    features = np.array([best_mu, e.stable_mu or 0., gamma, m/M,
        remaining_sdp_calls, c.max_sdp_calls,
        steps/c.max_steps, zero_cost_steps/c.max_zero_cost_steps], dtype=np.float32)
    return dict(nodes=nodes, edges=edges, node_mask=node_mask, operation_mask=operations,
                pair_mask=pairs, global_features=features, status=status,
                diagnostics=diagnostics, diagnostic_mask=diagnostic_mask)


class KuramotoEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, config: EnvironmentConfig | None = None, *,
                 evaluation_config: EvaluationConfig | None = None,
                 initial_template: PhaseTemplate | None = None, evaluator=None,
                 output: str | Path | None = None, render_mode: str | None = None):
        self.config = config or EnvironmentConfig()
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.render_mode = render_mode
        if evaluator is not None and evaluation_config is not None:
            raise ValueError("Supply evaluator or evaluation_config, not both")
        if isinstance(evaluator, CachedEvaluator):
            raise ValueError("The serial environment requires an uncached evaluator")
        self.evaluator = evaluator if evaluator is not None else TemplateEvaluator(
            evaluation_config or EvaluationConfig(max_clusters=self.config.max_clusters))
        if self.evaluator.config.max_clusters < self.config.max_clusters:
            raise ValueError("Evaluator max_clusters is smaller than environment max_clusters")
        m = min(self.config.max_clusters, max(self.config.min_clusters, 5))
        self.initial_template = initial_template if initial_template is not None else twisted_seed(m)
        self.output = Path(output) if output is not None else None
        self.action_space = action_space(self.config.max_clusters)
        M = self.config.max_clusters
        self.observation_space = spaces.Dict({
            "nodes": spaces.Box(-np.inf, np.inf, (M, 2), dtype=np.float32),
            "edges": spaces.Box(-1, 1, (M, M, 2), dtype=np.float32),
            "node_mask": spaces.MultiBinary(M),
            "operation_mask": spaces.MultiBinary(4),
            "pair_mask": spaces.MultiBinary((4, M, M)),
            "global_features": spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
            "status": spaces.MultiBinary(len(STATUSES)),
            "diagnostics": spaces.Box(-np.inf, np.inf, (2, 4), dtype=np.float32),
            "diagnostic_mask": spaces.MultiBinary((2, 4)),
        })
        self._session = None
        self._record = None
        self._active = False
        self._steps = self._zero_cost = self._episode = 0
        self._transition_log = None
        self._event_log = None
        self.last_result = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._finish("reset")
        options = {} if options is None else dict(options)
        if set(options) - {"template"}:
            raise ValueError("reset options supports only 'template'")
        template = options.get("template", self.initial_template)
        if not isinstance(template, PhaseTemplate):
            raise TypeError("reset template must be a PhaseTemplate")
        episode_seed = int(self.np_random.integers(0, 2**32))
        self.action_space.seed(episode_seed)
        destination = self.output / f"episode-{self._episode:06d}" if self.output is not None else None
        session = SearchSession([template], self.evaluator, seed=episode_seed,
            min_clusters=self.config.min_clusters, max_clusters=self.config.max_clusters,
            max_sdp_calls=self.config.max_sdp_calls, max_wall_seconds=self.config.max_wall_seconds,
            output=destination, deduplicate=False,
            search_config=dict(algorithm="gymnasium_serial", environment_schema_version=1, **asdict(self.config)))
        try:
            session.__enter__()
            self._record = session.evaluate_seeds()[0]
            if session.budget.sdp_calls > self.config.max_sdp_calls:
                raise RuntimeError("Evaluator exceeded the SDP budget")
            if destination is not None:
                self._transition_log = (destination / "transitions.jsonl").open("x", encoding="utf-8")
                self._event_log = (destination / "events.jsonl").open("x", encoding="utf-8")
        except BaseException:
            session.__exit__(None, None, None)
            for name in ("_transition_log", "_event_log"):
                handle = getattr(self, name)
                if handle is not None:
                    handle.close()
                    setattr(self, name, None)
            raise
        self._session = session
        self._steps = self._zero_cost = 0
        self._active = True
        self._episode += 1
        self.last_result = None
        self._event("reset", seed=episode_seed, state_id=self._record.id,
                    seed_sdp_calls=session.budget.sdp_calls, best_mu=self._best_mu(),
                    remaining_sdp_calls=session.budget.remaining())
        return self._observation(), self._info(session.budget.sdp_calls, "reset")

    def step(self, action):
        if not self._active:
            raise ResetNeeded("Call reset() before step() and after an episode ends")
        if not self.action_space.contains(action):
            raise ValueError("Action is outside action_space")
        action = dict(operation=int(action["operation"]), i=int(action["i"]), j=int(action["j"]),
                      parameters=np.asarray(action["parameters"], dtype=np.float32))
        session = self._session
        reason = session.budget.stop_reason()
        if reason is not None:
            return self._end_without_action(reason)
        source = self._record
        before_mu, before_calls = self._best_mu(), session.budget.sdp_calls
        proposal = decode_action(action, source.template, self.config)
        record = session.record(proposal, source.id, "gymnasium_serial")
        cost = session.budget.sdp_calls - before_calls
        if cost < 0 or session.budget.sdp_calls > self.config.max_sdp_calls:
            self._finish("evaluator_error")
            raise RuntimeError("Evaluator returned invalid SDP-call accounting")
        # Evaluated unstable/invalid states are part of the trajectory. A
        # rejected structural action leaves the current state unchanged.
        if record.evaluation is not None:
            self._record = record
        self._steps += 1
        self._zero_cost = self._zero_cost + 1 if cost == 0 else 0
        reward = float(self.config.reward_scale * (self._best_mu() - before_mu))
        terminated = bool(session.budget.remaining() <= 0)
        reason = "sdp_budget" if terminated else (
            "wall_time" if session.budget.stop_reason() == "wall_time" else
            "step_limit" if self._steps >= self.config.max_steps else
            "zero_cost_limit" if self._zero_cost >= self.config.max_zero_cost_steps else "")
        truncated = bool(reason and not terminated)
        observation = self._observation()
        info = self._info(cost, reason)
        info.update(record_id=record.id, disposition=record.disposition, message=record.message)
        if self._transition_log is not None:
            data = dict(schema_version=1, source_id=source.id, record_id=record.id,
                next_state_id=self._record.id,
                action={k: (v.tolist() if isinstance(v, np.ndarray) else int(v)) for k, v in action.items()},
                best_mu_before=before_mu, best_mu_after=self._best_mu(),
                remaining_calls_before=self.config.max_sdp_calls-before_calls,
                remaining_calls_after=session.budget.remaining(), sdp_calls=cost, reward=reward,
                terminated=terminated, truncated=truncated, stop_reason=reason)
            self._transition_log.write(json.dumps(data, allow_nan=False)+"\n")
            self._transition_log.flush()
        if terminated or truncated:
            self._finish(reason)
        return observation, reward, terminated, truncated, info

    def sample_action(self):
        """Uniform valid hierarchy sampler; uses only this env's seeded RNG."""
        if not self._active:
            raise ResetNeeded("Call reset before sampling a valid action")
        operations, nodes, pairs = action_masks(self._record.template, self.config)
        op = int(self.np_random.choice(np.flatnonzero(operations)))
        if op in (1, 3):
            choices = np.argwhere(pairs[op])
            i, j = (int(v) for v in choices[int(self.np_random.integers(len(choices)))])
        else:
            i = int(self.np_random.choice(np.flatnonzero(nodes)))
            j = 0
        return dict(operation=op, i=i, j=j,
                    parameters=self.np_random.uniform(-1, 1, 2).astype(np.float32))

    def _best_mu(self):
        best = self._session.archive.best
        return float(best.evaluation.stable_mu) if best is not None else 0.0

    def _observation(self):
        return build_observation(self._record.template, self._record.evaluation, self.config,
            best_mu=self._best_mu(), remaining_sdp_calls=self._session.budget.remaining(),
            steps=self._steps, zero_cost_steps=self._zero_cost)

    def _info(self, cost, reason):
        return dict(sdp_calls=int(cost), total_sdp_calls=self._session.budget.sdp_calls,
            remaining_sdp_calls=self._session.budget.remaining(), best_mu=self._best_mu(),
            status=self._record.evaluation.status, steps=self._steps, stop_reason=reason,
            state_id=self._record.id)

    def _event(self, event, **data):
        if self._event_log is not None:
            self._event_log.write(json.dumps(dict(schema_version=1, event=event, **data), allow_nan=False)+"\n")
            self._event_log.flush()

    def _end_without_action(self, reason):
        observation, info = self._observation(), self._info(0, reason)
        self._finish(reason)
        return observation, 0.0, reason == "sdp_budget", reason != "sdp_budget", info

    def _finish(self, reason):
        if self._session is None or not self._active:
            return
        try:
            self._event("episode_end", state_id=self._record.id, stop_reason=reason,
                        terminated=reason == "sdp_budget", truncated=reason != "sdp_budget",
                        total_sdp_calls=self._session.budget.sdp_calls, best_mu=self._best_mu())
            self.last_result = self._session.finish(reason)
        finally:
            self._session.__exit__(None, None, None)
            if self._transition_log is not None:
                self._transition_log.close()
                self._transition_log = None
            if self._event_log is not None:
                self._event_log.close()
                self._event_log = None
            self._active = False

    def render(self):
        if self.render_mode == "ansi" and self._record is not None:
            return (f"clusters={self._record.template.n_clusters} "
                    f"best_mu={self._best_mu():.10f} "
                    f"remaining_calls={self._session.budget.remaining()}")
        return None

    def close(self):
        self._finish("closed")
