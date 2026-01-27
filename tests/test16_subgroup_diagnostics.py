import numpy as np

from metrics_lie.diagnostics.subgroups import compute_group_sizes, group_indices


def test_subgroup_helpers():
    subgroup = np.array(["A", "A", "B", "B", "A"])
    groups = group_indices(subgroup)
    sizes = compute_group_sizes(subgroup)

    assert "A" in groups
    assert "B" in groups
    assert groups["A"].sum() == 3
    assert groups["B"].sum() == 2
    assert sizes["A"] == 3
    assert sizes["B"] == 2
    assert set(sizes.keys()) == {"A", "B"}

