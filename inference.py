from pathlib import Path
from typing import Literal, Optional
import joblib
from sentence_transformers import SentenceTransformer
import torch

SBERT_PATH = Path("models/models/sentence_transformer.model")
LOGREG_PATH = Path("models/models/classifier.joblib")

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
        encoder_dir: Path = SBERT_PATH,
        logreg_path: Path = LOGREG_PATH,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(str(encoder_dir), device=device)
        self.clf = joblib.load(logreg_path)

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
