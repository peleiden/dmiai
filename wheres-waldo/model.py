from abc import ABC, abstractmethod

import cv2
import pandas as pd
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

from data_utils import load_img

class WaldoModel(ABC):
    @abstractmethod
    def predict(self, img: np.ndarray) -> tuple[float, float]:
        pass

class TemplateMatcher(WaldoModel):
    def __init__(self, template_paths: list[str], scorer=cv2.TM_CCOEFF_NORMED, verbose=True):
        self.template_paths = template_paths
        self.templates = [load_img(p) for p in template_paths]
        self.scorer = scorer
        self.verbose = verbose
        self.scales = [*np.linspace(0.7, 0.9, 3), 1, *np.linspace(1.1, 2, 10)]
        if self.verbose:
            print("Templates", self.template_paths)
            print("Scales", [round(s, 3) for s in self.scales])

    def match(self, img: np.ndarray, template: np.ndarray) -> np.ndarray:
        return cv2.matchTemplate(img, template, self.scorer)

    def match_infer(self, img: np.ndarray, template: np.ndarray) -> tuple[tuple[int, int], float]:
        match = self.match(img, template)
        _, score, _, top_left = cv2.minMaxLoc(match)
        pos = (top_left[0] + template.shape[0]/2, top_left[1] + template.shape[1]/2)
        return pos, score

    def predict_scales(self, img: np.ndarray, template: np.ndarray) -> tuple[tuple[int, int], float]:
        scores, positions = list(), list()
        for scale in self.scales:
            template_r = cv2.resize(img, (int(template.shape[0]*scale), int(template.shape[1]*scale)))
            pos, score = self.match_infer(img, template_r)
            scores.append(score)
            positions.append(pos)
        i = np.argmax(scores)
        if self.verbose:
            print(f"Best size {self.scales[i]} w/ score {round(scores[i], 4)}")
        return positions[i], scores[i]

    def predict(self, img: np.ndarray) -> tuple[float, float]:
        locs, scores = zip(*[self.predict_scales(img, t) for t in self.templates])
        i = np.argmax(scores)
        if self.verbose:
            print(f"Best template {self.template_paths[i]} w/ score {round(scores[i], 4)}")
        return locs[i]

# class FaceNet(WaldoModel):
#     def __init__(self):



if __name__ == "__main__":
    df = pd.read_pickle("data/df.pkl")
    img = df[df.waldo].img.iloc[0]
    m = TemplateMatcher(["assets/templates/face.jpg", "assets/templates/torso.jpg"])
    # print(m.predict(img))
    mtcnn = MTCNN(64, min_face_size=10, keep_all=True)
    mtcnn(img)
    breakpoint()
