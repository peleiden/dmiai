from __future__ import annotations
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def setup_yolo_model():
    from darkflow.net.build import TFNet
    paths = {
        "pbLoad": "yolov2-wally.pb",
        "metaLoad": "yolov2-wally.meta"
    }
    return TFNet(paths)

def yolo_predict(model, img: np.ndarray) -> None | tuple[int, int]:
    results = model.return_predict(img)
    if not results:
        return 0, 0
    best = sorted(results, key=lambda x: x["confidence"])[-1]
    print(best["confidence"])
    x = (best["topleft"]["x"] + best["bottomright"]["x"]) / 2
    y = (best["topleft"]["y"] + best["bottomright"]["y"]) / 2
    return x, y

if __name__ == "__main__":
    from PIL import Image
    m = setup_model()
    print(
        yolo_predict(m, np.array(Image.open("data/waldo/1_1_1.jpg")))
    )
