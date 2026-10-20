import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from predict import image_to_features
from train import candidates


def test_image_to_features_shape_and_range(tmp_path):
    p = tmp_path / "d.png"
    Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8)).save(p)
    X = image_to_features(p)
    assert X.shape == (1, 64)
    assert 0 <= X.min() and X.max() <= 16


def test_every_candidate_beats_90_percent():
    d = load_digits()
    X_tr, X_te, y_tr, y_te = train_test_split(d.data, d.target, test_size=0.2, random_state=0)
    for name, (pipe, _) in candidates().items():
        assert pipe.fit(X_tr, y_tr).score(X_te, y_te) > 0.90, name
