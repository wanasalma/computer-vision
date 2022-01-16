from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

class FeatureExtractor:
    def __init__(self):

        self.base_model = InceptionResNetV2(weights="imagenet")
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer("avg_pool").output)
     

    def extract(self, img):
        height = 299
        width = 299
        img = img.resize((height, width)).convert("RGB")
        x = image.img_to_array(img)  # to np.array
        x = np.expand_dims(x, axis=0)  # (H, W, C) -> (1, H, W, C)
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096)
        return feature/np.linalg.norm(feature)  # normalize