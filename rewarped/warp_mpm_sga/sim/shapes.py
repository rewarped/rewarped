# from https://github.com/sizhe-li/DexDeform/blob/main/mpm/shapes.py

import numpy as np


def length(x):
    return np.sqrt(np.einsum('ij,ij->i', x, x) + 1e-14)


def compute_cylinder_sdf(h, r, half=False):
    def sdf_func(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        vec_xz = np.stack([x, z], axis=1)

        rh = np.array([[r, h]])
        d = np.abs(np.stack([length(vec_xz), y], axis=1)) - rh

        if half:
            half_mask = x >= 0
            d[~half_mask, 0] = length(np.stack([x[~half_mask], np.maximum(z[~half_mask] - r, 0)], axis=1))

        return np.minimum(np.maximum(d[:, 0], d[:, 1]), 0.0) + length(np.maximum(d, 0.0))

    return sdf_func


def box_particles(width):
    def sample_func(n_particles, rng):
        # p = (np.random.random((n_particles, 3)) * 2 - 1) * width
        p = (rng.uniform(0.0, 1.0, (n_particles, 3)) * 2 - 1) * width
        return p

    return sample_func


def rejection_sampling(init_pos, n_particles, sample_func, sdf_func, rng):
    p = np.ones((n_particles, 3)) * 5

    remain_cnt = n_particles  # how many left to sample
    while remain_cnt > 0:
        p_samples = sample_func(remain_cnt, rng)
        sdf_vals = sdf_func(p_samples)

        accept_map = sdf_vals <= 0
        accept_cnt = sum(accept_map)
        start = n_particles - remain_cnt
        p[start:start + accept_cnt] = p_samples[accept_map]
        remain_cnt -= accept_cnt
    assert np.all(p != 5)
    p = p + np.array(init_pos)

    return p
