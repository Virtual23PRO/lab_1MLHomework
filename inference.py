from pathlib import Path
from typing import Literal, Optional
import joblib
from sentence_transformers import SentenceTransformer
import torch

MODELS_DIR = Path("models")

Label = Literal["negative", "neutral", "positive"]
ID2LABEL: dict[int, Label] = {0: "negative", 1: "neutral", 2: "positive"}


def _find_encoder_dir(root: Path) -> Path:
    for p in root.iterdir():
        if p.is_dir() and (p / "config.json").exists():
            return p
    raise FileNotFoundError("There is no model")


def _find_logreg_file(root: Path) -> Path:
    for p in root.glob("*.joblib"):
        return p
    raise FileNotFoundError("There is no *.joblib")


class SentimentPipeline:
    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        encoder_dir: Optional[Path] = None,
        logreg_path: Optional[Path] = None,
    ):
        models_dir = models_dir.resolve()
        enc_dir = encoder_dir or _find_encoder_dir(models_dir)
        clf_path = logreg_path or _find_logreg_file(models_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(str(enc_dir), device=device)
        self.clf = joblib.load(clf_path)

    def predict(self, text: str) -> Label:
        emb = self.encoder.encode([text], convert_to_numpy=True)
        pred = int(self.clf.predict(emb)[0])
        return ID2LABEL[pred]


_pipeline: Optional[SentimentPipeline] = None


def get_pipeline() -> SentimentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SentimentPipeline()
    return _pipeline
