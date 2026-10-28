# Joint limiting SDP pilot

Best strict limiting density: 0.6929033189148635.
Best independently strict finite lift: 0.6922885138802887.
Comparison target: 0.6929077063695864; rigorous certification remains unavailable.

The calibration passed before the sweep. Original two-stage algorithms are unchanged.
Paired cells share masses, normalized offsets, labels and support, with a two-all-conic ceiling and zero full SDPs.
Early infeasibility/necessary-screen failure spends fewer calls; matched_actual_all_conic identifies spent ceilings.
Joint repair-tree inward conditions, explicit box buffers, and fixed nonuniform-anchor entries are additional inner restrictions.
Old controls optimize their supported unrestricted densities on the same mask; these differences prevent a superiority claim.
Their R solve requests its effective-soft margin; the joint model controls the full fixed-gauge block. All settings are saved.
This is not a matched wall-time experiment. No margin is ranked across cluster sizes.

| Method | Cells | Strict limits | Best strict mu0 | All conic calls | Wall seconds |
|---|---:|---:|---:|---:|---:|
| joint | 30 | 8 | 0.69286966467753 | 38 | 0.780 |
| lp_then_r | 30 | 3 | 0.47215746539391446 | 34 | 0.212 |
| sdp_then_r | 30 | 6 | 0.6923949582255815 | 36 | 0.428 |

Sweep solver costs: {'joint_sdp': 1048, 'finite_tree_linear': 30, 'full_density_sdp': 0, 'full_margin_sdp': 0, 'high_precision_audit': 10}.
Sweep outcomes: {'joint_asymptotic_strict': 416, 'uncertain': 158, 'conditional_infeasible': 473, 'joint_unstable': 1}.
Every proposal, selected parent, partial weak-mode prediction and re-solve is recorded.
A limiting improvement is not a finite witness, and a numerical witness is not a proved lower bound.
