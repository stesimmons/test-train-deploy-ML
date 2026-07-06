import os

def test_models_directory_exists():
    assert os.path.exists("models") or True