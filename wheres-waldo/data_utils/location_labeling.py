import os
from glob import glob as glob # glob
import json

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

def main():
    out = dict()
    for fp in glob("data/waldo/*"):
        _, name = os.path.split(fp)
        img = np.array(Image.open(fp))
        plt.imshow(img)
        plt.show()
        out[name] = input(name + ":")

    print(json.dumps(out, indent=4))

if __name__ == "__main__":
    main()
