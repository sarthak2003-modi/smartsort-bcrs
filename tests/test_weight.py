"""Weight parsing and band checks."""

from smartsort import config
from smartsort.sensors.weight import ManualWeightEntry, in_band, parse_weight_grams


def test_parse_ok():
    r = parse_weight_grams("18")
    assert r.ok and r.grams == 18.0


def test_parse_out_of_entry_range():
    r = parse_weight_grams(6000)
    assert not r.ok


def test_parse_non_numeric():
    r = parse_weight_grams("abc")
    assert not r.ok


def test_pet_band():
    assert in_band(18.0, config.WEIGHT_BAND["pet-bottle"])
    assert not in_band(100.0, config.WEIGHT_BAND["pet-bottle"])


def test_can_band():
    assert in_band(15.0, config.WEIGHT_BAND["aluminium-can"])
    assert not in_band(40.0, config.WEIGHT_BAND["aluminium-can"])


def test_manual_entry():
    assert ManualWeightEntry(12.5).read().ok
