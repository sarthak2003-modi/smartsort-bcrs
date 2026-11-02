from pathlib import Path

from smartsort.capture import synthetic_chute_image
from smartsort.metrics import TrialCase, evaluate_trials
from smartsort.models.classifier import MockClassifier
from smartsort.oplog import OperatorLog
from smartsort.pipeline import evaluate_return


def test_oplog_roundtrip(tmp_path: Path):
    img = synthetic_chute_image()
    mock = MockClassifier("pet-bottle", 0.9)
    d = evaluate_return(
        img, 18.0, classify_fn=lambda i: (mock.predict(i).category, mock.predict(i).confidence)
    )
    log = OperatorLog(tmp_path / "op.jsonl", tmp_path / "op.csv")
    entry = log.append(d)
    rows = log.read_all()
    assert len(rows) == 1
    assert rows[0]["id"] == entry["id"]
    assert (tmp_path / "op.csv").exists()


def test_metrics_targets_structure():
    cases = [
        TrialCase(True, True, 0.1),
        TrialCase(False, False, 0.2),
        TrialCase(False, True, 0.1),  # FP fraud paid
        TrialCase(True, False, 0.1),  # FN valid held
    ]
    report = evaluate_trials(cases)
    assert report.n == 4
    assert report.accuracy == 0.5
    assert report.fpr == 0.5
    assert report.fnr == 0.5
    assert "accuracy" in report.targets
