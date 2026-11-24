import numpy as np
from imputasi.masks import block_mask, random_mask


def test_random_ratio_exact():
    m = random_mask((1000, 7), 0.3, seed=0)
    assert m.sum() == round(0.3 * 7000)


def test_random_reproducible_and_seed_differs():
    a, b, c = random_mask((500, 3), 0.2, 1), random_mask((500, 3), 0.2, 1), random_mask((500, 3), 0.2, 2)
    assert (a == b).all() and not (a == c).all()


def _runs(col):
    d = np.diff(np.concatenate([[0], col.astype(int), [0]]))
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def test_block_lengths_and_ratio_independent():
    m = block_mask((3484, 7), 72, 0.2, seed=0, simultaneous=False)
    for k in range(7):
        n_blocks = round(0.2 * 3484 / 72)
        assert m[:, k].sum() == n_blocks * 72          # tak tumpang-tindih
        assert (_runs(m[:, k]) % 72 == 0).all()         # run = kelipatan 72 (blok bisa menempel)
    assert not (m[:, 0] == m[:, 1]).all()               # posisi independen antar variabel


def test_block_simultaneous_same_across_columns():
    m = block_mask((3484, 7), 24, 0.2, seed=3, simultaneous=True)
    assert (m == m[:, [0]]).all()
