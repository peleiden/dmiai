import os
from glob import glob as glob # glob
import json

from PIL import Image
import numpy as np

import pandas as pd

from assets import ASSETPATH

def load_img(fp: str) -> np.ndarray:
    return np.array(Image.open(fp))

def build_df(p: str="data") -> pd.DataFrame:
    d = {
        "name": list(),
        "img": list(),
        "waldo": list(),
        "pos": list(),
    }
    with open(os.path.join(ASSETPATH, "location.json")) as f:
        positions = json.load(f)

    for fp in glob(os.path.join(p, "notwaldo") + "/*"):
        d["name"].append(os.path.split(fp)[-1])
        d["img"].append(load_img(fp))
        d["waldo"].append(False)
        d["pos"].append(None)

    for fp in glob(os.path.join(p, "waldo") + "/*"):
        name = os.path.split(fp)[-1]
        d["name"].append(name)
        d["img"].append(load_img(fp))
        d["waldo"].append(True)
        pos = positions[name].strip().split(",")
        d["pos"].append((float(pos[0]), float(pos[1])))
    return pd.DataFrame(d)

if __name__ == "__main__":
    build_df().to_pickle("data/df.pkl")
