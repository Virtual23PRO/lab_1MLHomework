from pathlib import Path
import joblib
from inference import LOGREG_PATH


def test_model_joblib_loads_without_errors():
    assert isinstance(LOGREG_PATH, Path)
    assert LOGREG_PATH.exists(), f"Brak pliku modelu: {LOGREG_PATH}"

    clf = joblib.load(LOGREG_PATH)
    assert hasattr(clf, "predict")
