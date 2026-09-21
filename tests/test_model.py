import os
import pytest
import tensorflow as tf
import numpy as np


def test_model_file_exists():
    """Check that the production model exists on disk"""
    model_path = "model/model.keras"
    assert os.path.exists(model_path), f"Model file not found at {model_path}"


def test_model_can_load():
    """Verify that TensorFlow can load the model without errors"""
    model = tf.keras.models.load_model("model/model.keras")
    assert model is not None


def test_model_output_shape():
    """Test model inference on a dummy image to ensure it outputs 4 classes"""
    model = tf.keras.models.load_model("model/model.keras")
    
    # Create a blank dummy image of shape (1, 224, 224, 3)
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    
    predictions = model.predict(dummy_input)
    
    # Must output 4 probabilities (one for each tumor category)
    assert predictions.shape == (1, 4)