import random

import numpy as np
import pandas as pd


PIECES = 5

def generate_imgs(df: pd.DataFrame, N: int) -> pd.DataFrame:
    dtype = df.img[0].dtype
    in_shape = df.img[0].shape # Assume everything is same size and quadratic
    out_shape = (in_shape[0]*PIECES, in_shape[1]*PIECES, in_shape[2])
    out_data = {
        "img": list(),
        "pos": list(),
    }

    waldo_idx = list(df[df.waldo].index)
    notwaldo_idx = list(df[~df.waldo].index)

    random.shuffle(waldo_idx)
    if (to_sample := N - len(waldo_idx)) > 0:
        waldo_idx = [*waldo_idx, *random.sample(waldo_idx, to_sample)]
    else:
        waldo_idx = waldo_idx[:N]

    for i in waldo_idx:
        img_idcs = [i, *random.sample(notwaldo_idx, PIECES**2-1)]
        random.shuffle(img_idcs)
        img_iter = iter(img_idcs)

        out_img = np.zeros(out_shape, dtype=dtype)
        for j in range(PIECES):
            for k in range(PIECES):
                idx = next(img_iter)
                if idx == i:
                    local_pos = df.pos[i]
                    out_data["pos"].append((j*in_shape[0]+local_pos[0], k*in_shape[0]+local_pos[1]))
                out_img[j*in_shape[0]:(j+1)*in_shape[0], k*in_shape[0]:(k+1)*in_shape[0]] = df.img[idx]
        out_data["img"].append(out_img)
    return pd.DataFrame(out_data)

if __name__ == "__main__":
    df = pd.read_pickle("data/df.pkl")
    import matplotlib.pyplot as plt
    plt.imshow(generate_imgs(df, 1)["img"][0])
    plt.show()
