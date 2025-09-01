from tket_eccs import nam_6_3, clifford_t_6_3


def test_load_nam_rewriter():
    rewriter = nam_6_3()
    assert rewriter is not None


def test_load_clifford_t_rewriter():
    rewriter = clifford_t_6_3()
    assert rewriter is not None
