from dataclasses import replace

from kuramoto_template.asymptotic_search.families import s7, split_candidate
from kuramoto_template.asymptotic_search.models import candidate_data
from kuramoto_template.asymptotic_search.persistence import _shortlist


def test_shortlist_covers_sizes_before_extra_masks():
    c = s7()
    mask, p = c.allowed_mask.copy(), c.p0.copy()
    mask[0, 3] = mask[3, 0] = False
    p[0, 3] = p[3, 0] = 0
    candidates = [c, replace(c, allowed_mask=mask, p0=p)]
    for size in (8, 9, 10, 12):
        split = c
        while split.m < size:
            split = split_candidate(split, 0, .5, .1)
        candidates.append(split)
    # Isolate structural selection from the independently tested numerical screen.
    records = [dict(candidate=candidate_data(candidate), metrics=dict(
        status="stage1_screened", mu0=.69, dipole_slacks=(.01, .01)))
        for candidate in candidates]
    selected = _shortlist(records, 5)
    assert {len(records[i]["candidate"]["x"]) for i in selected} == {7, 8, 9, 10, 12}
