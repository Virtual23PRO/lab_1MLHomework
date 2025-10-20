import pytest
from inference import get_pipeline, ID2LABEL

ALLOWED = set(ID2LABEL.values())


@pytest.mark.parametrize(
    "text",
    [
        "I love play football!",
        "I want to eat an apple.",
        "I hate boring people.",
    ],
)
def test_pipeline_predict_works_for_multiple_examples(text: str):
    pipe = get_pipeline()
    label = pipe.predict(text)
    assert label in ALLOWED
