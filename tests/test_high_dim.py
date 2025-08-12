import numpy as np

from sheshe.sheshe import generate_directions, rays_count_auto


def test_generate_directions_high_dimension():
    dim = 100
    base_2d = 8
    max_subspaces = 10
    D = generate_directions(dim, base_2d, random_state=0, max_subspaces=max_subspaces)
    n_global = rays_count_auto(dim, base_2d)
    n_local = rays_count_auto(3, base_2d)
    assert D.shape == (n_global + max_subspaces * n_local, dim)
    local_rows = D[n_global:]
    supports = set()
    for i in range(0, local_rows.shape[0], n_local):
        block = local_rows[i:i + n_local]
        support = tuple(np.where(np.any(np.abs(block) > 1e-8, axis=0))[0])
        assert len(support) == 3
        supports.add(support)
    assert len(supports) == max_subspaces
