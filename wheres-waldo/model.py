from abc import ABC, abstractmethod

import cv2
import pandas as pd
import numpy as np

from data_utils import load_img

class WaldoModel(ABC):
    @abstractmethod
    def predict(self, img: np.ndarray) -> tuple[float, float]:
        pass

class TemplateMatcher(WaldoModel):
    def __init__(self, template_paths: list[str], scorer=cv2.TM_CCOEFF_NORMED):
        self.templates = [load_img(p) for p in template_paths]
        self.scorer = scorer

    def match(self, img: np.ndarray, template: np.ndarray) -> np.ndarray:
        return cv2.matchTemplate(img, template, self.scorer)

    def best_match(self, img: np.ndarray, template: np.ndarray) -> tuple[float, float]:
        match = self.match(img, template)
        _, score, _, top_left = cv2.minMaxLoc(match)
        return (top_left[0] + template.shape[0]/2, top_left[1] + template.shape[1]/2)

    def predict(self, img: np.ndarray) -> tuple[float, float]:
        locs = [self.best_match(img, t) for t in self.templates]
        return locs[0]

if __name__ == "__main__":
    df = pd.read_pickle("data/df.pkl")
    imgs = df[df.waldo].img
    m = TemplateMatcher(["assets/templates/face.jpg"])
    loc = m.predict(imgs.iloc[0])
    breakpoint()
