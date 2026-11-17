"""Five workloads shown in the paper figures and human-readable report.

Retain historical measurements in summary.json. Original W6 becomes W5;
select by workload metadata so the mapping does not depend on array order.
"""
WORKLOADS = (
    (64, 64, 1, False),
    (64, 128, 16, False),
    (512, 128, 4, False),
    (1500, 128, 4, False),
    (512, 128, 4, True),
)


def selected_performance(rows):
    eligible = [r for r in rows if r["workload"][3] == 0]
    lookup = {(*r["workload"][:3], r["sampled"]): r for r in eligible}
    assert len(eligible) == len(lookup) == len(WORKLOADS)
    assert set(lookup) == set(WORKLOADS)
    return [lookup[key] for key in WORKLOADS]
