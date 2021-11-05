from __future__ import annotations
import tensorflow as tf
import numpy as np

PATH = "frozen_inference_graph.pb"

def setup_magajna_model():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def magajna_predict(model, img: np.ndarray) -> tuple[int, int]:
    with model.as_default():
        with tf.Session(graph=model) as sess:
            image_tensor    = model.get_tensor_by_name('image_tensor:0')
            boxes           = model.get_tensor_by_name('detection_boxes:0')
            scores          = model.get_tensor_by_name('detection_scores:0')
            classes         = model.get_tensor_by_name('detection_classes:0')
            num_detections  = model.get_tensor_by_name('num_detections:0')

            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: np.expand_dims(img, axis=0)}
            )
    print(scores[0][0])
    xmin, xmax, ymin, ymax = boxes[0][0]
    x = (xmin+xmax)/2 * img.shape[0]
    y = (ymin+ymax)/2 * img.shape[1]
    return x, y

if __name__ == "__main__":
    from PIL import Image
    m = setup_model()
    print(
        magajna_predict(m, np.array(Image.open("data/waldo/1_1_1.jpg")))
    )
