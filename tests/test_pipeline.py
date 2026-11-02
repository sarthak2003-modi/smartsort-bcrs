"""Pipeline evaluate_return with a mocked classifier (no weights / GPU)."""

from smartsort.capture import synthetic_chute_image
from smartsort.models.classifier import MockClassifier
from smartsort.pipeline import evaluate_return
from smartsort.sensors.material import MaterialReading


def _img():
    return synthetic_chute_image(seed=1)


def _clf(category: str, conf: float):
    mock = MockClassifier(category=category, confidence=conf)
    return lambda img: (mock.predict(img).category, mock.predict(img).confidence)


def test_pay_eligible_pet():
    d = evaluate_return(_img(), 18.0, classify_fn=_clf("pet-bottle", 0.92))
    assert d.action == "PAY"
    assert d.refund_cents == 10
    assert d.eligible is True


def test_hold_low_confidence():
    d = evaluate_return(_img(), 18.0, classify_fn=_clf("pet-bottle", 0.2))
    assert d.action == "HOLD"
    assert "low vision confidence" in d.reasons[0]


def test_hold_ineligible_rubbish():
    d = evaluate_return(_img(), 18.0, classify_fn=_clf("rubbish", 0.95))
    assert d.action == "HOLD"
    assert any("ineligible" in r for r in d.reasons)


def test_hold_weight_mismatch():
    d = evaluate_return(_img(), 200.0, classify_fn=_clf("pet-bottle", 0.95))
    assert d.action == "HOLD"
    assert "weight mismatch" in d.reasons


def test_hold_fusion_too_low():
    # In-band + eligible, but ACCEPT_MIN raised so fused score fails the gate.
    d = evaluate_return(
        _img(),
        18.0,
        classify_fn=_clf("pet-bottle", 0.60),
        accept_min=0.95,
    )
    assert d.action == "HOLD"
    assert any("fusion" in r for r in d.reasons)


def test_aluminium_pay():
    d = evaluate_return(_img(), 15.0, classify_fn=_clf("aluminium-can", 0.9))
    assert d.action == "PAY"


def test_material_mismatch_hold():
    mat = MaterialReading(material="paper", confidence=0.95, source="test", ok=True)
    d = evaluate_return(
        _img(),
        18.0,
        classify_fn=_clf("pet-bottle", 0.95),
        material_reading=mat,
    )
    assert d.action == "HOLD"


def test_deck_alias_pet():
    d = evaluate_return(_img(), 20.0, classify_fn=lambda img: ("pet", 0.9))
    assert d.category == "pet-bottle"
    assert d.action == "PAY"
