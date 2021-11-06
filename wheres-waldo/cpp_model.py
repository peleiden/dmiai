import numpy as np
from wallyfinder import WallyFinder

def setup_cpp_model():
    return WallyFinder()

def cpp_predict(model, img):
    results = model(img)
    if not results:
        return 0, 0
    best = sorted(results, key=lambda x: x["confidence"])[-1]
    print(best["confidence"])
    x = int(round((best["xmax"] + best["xmin"]) / 2))
    y = int(round((best["ymax"] + best["ymin"]) / 2))
    return x, y

if __name__ == "__main__":
    from PIL import Image
    m = setup_cpp_model()
    print(
        cpp_predictt(m, np.array(Image.open("data/waldo/1_1_1.jpg")))
    )
