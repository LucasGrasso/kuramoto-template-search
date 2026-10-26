"""Supplementary solver ledger; full template calls remain SearchSession calls."""

from dataclasses import dataclass, field
from time import perf_counter


class CallLimit(RuntimeError):
    pass


@dataclass
class SolverLedger:
    limits: dict[str, int] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def remaining(self, kind):
        return self.limits.get(kind, 2**63-1) - self.counts.get(kind, 0)

    def invoke(self, kind, callback):
        if self.remaining(kind) <= 0:
            raise CallLimit(kind)
        self.counts[kind] = self.counts.get(kind, 0) + 1
        event = dict(kind=kind, status="interrupted", elapsed_seconds=0.)
        self.events.append(event)
        started = perf_counter()
        try:
            result = callback()
            event["status"] = str(getattr(result, "status", "completed"))
            if getattr(result,"message",""):
                event["message"] = result.message
            return result
        except BaseException as exc:
            event["status"] = "failed" if isinstance(exc,Exception) else "interrupted"
            event["message"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            event["elapsed_seconds"] = perf_counter()-started


def solve(problem, kind, numerics, ledger=None):
    ledger = ledger if ledger is not None else SolverLedger()

    def call():
        try:
            problem.solve(solver=numerics.solver, **numerics.solver_options)
        except (KeyboardInterrupt, SystemExit, GeneratorExit, MemoryError):
            raise
        except BaseException as exc:
            # Clarabel's native panic may inherit directly from BaseException.
            return type("SolveStatus", (), dict(status="solver_error", message=str(exc)))()
        return type("SolveStatus", (), dict(status=problem.status or "unknown", message=""))()

    return ledger.invoke(kind, call)
