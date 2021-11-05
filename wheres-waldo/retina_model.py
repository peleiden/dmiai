from __future__ import annotations
import numpy as np
import tensorflow as tf
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

PATH = "weights.h5"

global graph
graph = tf.get_default_graph()

def setup_retina_model():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(sess)
    return models.load_model(PATH, backbone_name="resnet50")

def retina_predict(model, img: np.ndarray) -> tuple[int, int]:
    with graph.as_default():
        image = preprocess_image(img)
        #image, scale = resize_image(image, min_side=1800, max_side=3000)
        image, scale = resize_image(image, min_side=1500, max_side=3000)
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        xmin, xmax, ymin, ymax = boxes[0][0]
        print(scores[0][0])
        x = int(round((xmin+xmax)/2))
        y = int(round((ymin+ymax)/2))
        return x, y

if __name__ == "__main__":
    from PIL import Image
    m = setup_retina_model()
    print(
        retina_predict(m, np.array(Image.open("data/waldo/1_1_1.jpg")))
    )
