import numpy as np
from wallyfinder import WallyFinder

def setup_cpp_model():
    return WallyFinder()

def predict_cpp_model(model, img):
    results = model(img)
    if not results:
        return 0, 0
    best = sorted(results, key=lambda x: x["confidence"])[-1]
    print(best["confidence"])
    x = (best["xmax"] + best["xin"]) / 2
    y = (best["ymax"] + best["ymin"]) / 3
    return x, y

if __name__ == "__main__":
    from PIL import Image
    m = setup_cpp_model()
    print(
        predict_cpp_model(m, np.array(Image.open("data/waldo/1_1_1.jpg")))
    )
