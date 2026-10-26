"""Fixed Gymnasium spaces and a deterministic mutation decoder.

Operation IDs: split=0, merge=1, perturb=2, resize=3. Unused action
arguments are ignored, not sampled internally; every mutation is replayable.
"""
import numpy as np
from gymnasium import spaces
from kuramoto_template.mutations import split_cluster, merge_clusters, perturb_phase, resize_clusters
from kuramoto_template.search.infrastructure import MutationProposal
from kuramoto_template.template import PhaseTemplate
from .config import EnvironmentConfig

OPERATIONS = ("split", "merge", "perturb", "resize")


def action_space(max_clusters: int):
    return spaces.Dict({
        "operation": spaces.Discrete(4),
        "i": spaces.Discrete(max_clusters),
        "j": spaces.Discrete(max_clusters),
        "parameters": spaces.Box(-1, 1, (2,), dtype=np.float32),
    })


def action_masks(template: PhaseTemplate, config: EnvironmentConfig):
    m, M = template.n_clusters, config.max_clusters
    pairs = np.zeros((4, M, M), dtype=np.int8)
    pairs[3, :m, :m] = 1
    np.fill_diagonal(pairs[3], 0)
    if m > config.min_clusters:
        for i in range(m):
            for j in range(i+1, m):
                mass = template.x[i] + template.x[j]
                resultant = template.x[i]*np.exp(1j*template.phi[i]) + template.x[j]*np.exp(1j*template.phi[j])
                pairs[1, i, j] = abs(resultant) > 1e-12*mass
    operations = np.array([m < M, bool(pairs[1].any()), True, True], dtype=np.int8)
    nodes = np.zeros(M, dtype=np.int8)
    nodes[:m] = 1
    return operations, nodes, pairs


def decode_action(action: dict, template: PhaseTemplate, config: EnvironmentConfig) -> MutationProposal:
    """Decode an in-space action; masked actions are recorded as rejections."""
    op, i, j = (int(action[k]) for k in ("operation", "i", "j"))
    p, q = (float(v) for v in action["parameters"])
    operations, nodes, pairs = action_masks(template, config)
    if not operations[op] or not nodes[i] or (op in (1, 3) and not pairs[op, i, j]):
        return MutationProposal(OPERATIONS[op], {}, message="Action selects a masked operation, node or pair")
    if op == 0:
        ratio = .2 + .6*(p+1)/2
        separation = (1 if q >= 0 else -1)*np.exp(
            np.log(config.separation_min) + abs(q)*np.log(config.separation_max/config.separation_min))
        parameters = dict(index=i, ratio=ratio, separation=float(separation))
        build = lambda: split_cluster(template, **parameters)
    elif op == 1:
        parameters = dict(i=i, j=j)
        build = lambda: merge_clusters(template, **parameters)
    elif op == 2:
        parameters = dict(index=i, delta=p*config.phase_scale)
        build = lambda: perturb_phase(template, **parameters)
    else:
        fraction = config.resize_fraction*(.01+.99*(p+1)/2)
        parameters = dict(source=i, target=j, amount=float(template.x[i]*fraction))
        build = lambda: resize_clusters(template, **parameters)
    try:
        return MutationProposal(OPERATIONS[op], parameters, build())
    except (ValueError, FloatingPointError) as exc:
        return MutationProposal(OPERATIONS[op], parameters, message=f"{type(exc).__name__}: {exc}")
