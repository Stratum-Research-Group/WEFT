"""
Exact algebraic verification of the Boolean single-bit uncrossing identity:
  SWAP = I - s s^T
Verifies exact matrix and truth-assignment identity on all 16 states.
Note: In normalized quantum notation with |s> = (|01> - |10>)/sqrt(2),
the projector P = |s><s| has P^2 = P and SWAP = I - 2|s><s|.
In unnormalized integer form, s = [0, 1, -1, 0]^T has s^T s = 2,
yielding SWAP = I - s s^T directly over integers without square roots.
"""

import itertools
import numpy as np


def s(p: bool, q: bool) -> int:
    """Unnormalized singlet coefficient: s(p,q) = (−1)^p [p ≠ q]"""
    if p == q:
        return 0
    return -1 if p else 1


def X1(x: bool, y: bool, u: bool, v: bool) -> int:
    """Crossing (SWAP): [x = v][y = u]"""
    return 1 if (x == v and y == u) else 0


def I1(x: bool, y: bool, u: bool, v: bool) -> int:
    """Uncrossed term (Identity): [x = u][y = v]"""
    return 1 if (x == u and y == v) else 0


def test_single_bit():
    """Verify X1(x, y, u, v) == I1(x, y, u, v) - s(x, y) * s(u, v) for all 16 states."""
    passed = 0
    for x in [False, True]:
        for y in [False, True]:
            for u in [False, True]:
                for v in [False, True]:
                    lhs = X1(x, y, u, v)
                    rhs = I1(x, y, u, v) - s(x, y) * s(u, v)
                    assert lhs == rhs, f"Mismatch at ({x}, {y}, {u}, {v})"
                    passed += 1
    assert passed == 16, f"Expected 16 checks, got {passed}"


def tensor_representation():
    """Show the tensor as a 4x4 matrix mapping (x, y) -> (u, v)."""
    # Basis: (0,0), (0,1), (1,0), (1,1)
    basis = [(False, False), (False, True), (True, False), (True, True)]

    # Crossing matrix (SWAP operator on the state space)
    M_cross = np.zeros((4, 4), dtype=int)
    M_I = np.zeros((4, 4), dtype=int)
    M_singlet = np.zeros((4, 4), dtype=int)

    for i, (x, y) in enumerate(basis):
        for j, (u, v) in enumerate(basis):
            M_cross[j, i] = X1(x, y, u, v)
            M_I[j, i] = I1(x, y, u, v)
            M_singlet[j, i] = s(x, y) * s(u, v)

    assert np.array_equal(M_cross, M_I - M_singlet)

    # Normalized singlet check: s_norm = s / sqrt(2)
    s_vec = np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2.0)
    P_norm = np.outer(s_vec, s_vec)
    # Projector idempotence: P^2 = P
    assert np.allclose(P_norm @ P_norm, P_norm)
    # Normalized SWAP: I - 2*P
    SWAP_norm = np.eye(4) - 2.0 * P_norm
    assert np.allclose(SWAP_norm, M_cross)

    return M_cross, M_I, M_singlet


if __name__ == "__main__":
    test_single_bit()
    print("Verified single-bit uncrossing identity on all 16 truth assignments.")
    cross_mat, i_mat, singlet_mat = tensor_representation()
    print("M_cross (SWAP):")
    print(cross_mat)
    print("M_I (Identity):")
    print(i_mat)
    print("M_singlet (Unnormalized outer product s s^T, trace=2):")
    print(singlet_mat)
    print(f"Rank of M_singlet: {np.linalg.matrix_rank(singlet_mat)}")
