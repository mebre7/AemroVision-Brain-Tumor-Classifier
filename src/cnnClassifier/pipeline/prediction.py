import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load the model
        model_path = Path("artifacts/training/model.keras")
        model = load_model(model_path)

        # Prepare the image for prediction = Preprocessing the image
        image_name = self.filename
        test_image = image.load_img(image_name, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = tf.expand_dims(test_image, 0)
        test_image = test_image / 255.0

        # Run inference
        predictions = model.predict(test_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.amax(predictions))

        # Class labels mapped to flow_from_directory alphabetical sorting order
        class_mapping = {
            0: "Glioma Tumor",
            1: "Meningioma Tumor",
            2: "No Tumor",
            3: "Pituitary Tumor"
        }

        result_label = class_mapping.get(predicted_class_index, "Unknown Class")
        return [{"prediction": result_label, "confidence": round(confidence * 100, 2)}]
