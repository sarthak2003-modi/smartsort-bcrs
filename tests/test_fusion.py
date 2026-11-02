"""Unit tests for weight_fit and fuse."""

from smartsort.fusion import fuse, weight_fit


def test_weight_fit_inside_plateau():
    assert weight_fit(39.0, (8.0, 70.0)) == 1.0
    assert weight_fit(8.0, (8.0, 70.0)) == 1.0
    assert weight_fit(70.0, (8.0, 70.0)) == 1.0
    assert weight_fit(18.0, (8.0, 70.0)) == 1.0


def test_weight_fit_outside():
    assert weight_fit(5.0, (8.0, 70.0)) == 0.0
    assert weight_fit(100.0, (8.0, 70.0)) == 0.0


def test_weight_fit_soft_edge():
    # 2 g outside low edge with soft_edge=4 → 0.5
    assert abs(weight_fit(6.0, (8.0, 70.0), soft_edge=4.0) - 0.5) < 1e-9


def test_fuse_two_signal():
    score = fuse(1.0, 1.0, material_score=None, w_cv=0.6, w_weight=0.4)
    assert abs(score - 1.0) < 1e-9


def test_fuse_blended():
    # 0.6*0.8 + 0.4*1.0 = 0.88
    assert abs(fuse(0.8, 1.0) - 0.88) < 1e-9


def test_fuse_with_material_renorm():
    score = fuse(1.0, 1.0, material_score=1.0, w_cv=0.6, w_weight=0.4, w_material=0.15)
    assert abs(score - 1.0) < 1e-9
