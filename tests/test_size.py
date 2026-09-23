import pytest

from catalog import classify_model_size


def test_one_feature_is_tiny():
    assert classify_model_size(1) == "tiny"

def test_zero_features_is_invalid():
    with pytest.raises(ValueError):
        classify_model_size(0)


def test_five_features_is_tiny():
    assert classify_model_size(5) == "tiny"

def test_six_features_is_small():
    assert classify_model_size(6) == "small"

def test_15_features_is_small():
    assert classify_model_size(15) == "small"

def test_16_features_is_medium():
    assert classify_model_size(16) == "medium"

def test_30_features_is_medium():
    assert classify_model_size(30) == "medium"

def test_31_features_is_small():
    assert classify_model_size(31) == "large"

def test_40_features_is_small():
    assert classify_model_size(40) == "large"
