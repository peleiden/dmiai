import numpy as np

import pandas as pd

from model import WaldoModel


def evaluate(df: pd.DataFrame, model: WaldoModel):
    preds, errors = list(), list()
    for img, truth in zip(df.img, df.pos):
        pred = model.predict(img)
        errors.append((pred[0]-truth[0])**2 + (pred[1]-truth[1])**2)
        preds.append(pred)
    print("Mean error", np.mean(errors))
    return preds, errors
