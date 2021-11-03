from abc import ABC, abstractmethod

import cv2
import pandas as pd
import numpy as np

from skimage.color import rgb2hsv
from skimage.color.colorconv import rgb2gray
from skimage.morphology import erosion, dilation, opening, closing
from scipy.ndimage import median_filter
from skimage.measure import label, regionprops, regionprops_table
from PIL import Image

from data_utils import load_img

class WaldoModel(ABC):
    @abstractmethod
    def predict(self, img: np.ndarray) -> tuple[float, float]:
        pass

class TemplateMatcher(WaldoModel):
    def __init__(self, template_paths: list[str], scorer=cv2.TM_CCOEFF_NORMED, verbose=True):
        self.template_paths = template_paths
        self.templates = [rgb2gray(load_img(p)).astype(np.uint8)*255 for p in template_paths]

        self.scorer = scorer
        self.verbose = verbose
        self.scales = [*np.linspace(0.7, 0.9, 3), 1, *np.linspace(1.1, 2, 10)]
        self.angles = [-45, -45/2, 0, 45/2, 45]
        if self.verbose:
            print("Templates", self.template_paths)
            print("Scales", [round(s, 3) for s in self.scales])
            print("Angles", [round(a, 3) for a in self.angles])

    def preprocess_input(self, img: np.ndarray) -> np.ndarray:
        return (rgb2hsv(img)[:, :, 2] < 0.5).astype(np.uint8)*255

    def match(self, img: np.ndarray, template: np.ndarray) -> np.ndarray:
        return cv2.matchTemplate(img, template, self.scorer)

    def match_infer(self, img: np.ndarray, template: np.ndarray) -> tuple[tuple[int, int], float]:
        match = self.match(img, template)
        _, score, _, top_left = cv2.minMaxLoc(match)
        pos = (top_left[0] + template.shape[0]/2, top_left[1] + template.shape[1]/2)
        return pos, score

    def candidates(self, img: np.ndarray) -> pd.DataFrame:
        img = self.preprocess_input(img)
        o = {
            "pos":       list(),
            "score":     list(),
            "template":  list(),
            "transform": list(),
        }
        for i, t in enumerate(self.templates):
            pos, scores, transformations = self.predict_transforms(img, t)
            o["pos"].extend(pos)
            o["score"].extend(scores)
            o["transform"].extend(transformations)
            o["template"].extend([self.template_paths[i]]*len(pos))

        return pd.DataFrame(o)

    def predict_transforms(self, img: np.ndarray, template: np.ndarray) -> tuple[list[tuple[int, int]], list[float], list[str]]:
        positions, scores, transformations = list(), list(), list()
        for scale in self.scales:
            template_s = np.array(Image.fromarray(template).resize(
                (int(template.shape[1]*scale), int(template.shape[0]*scale)), resample=Image.LANCZOS
            ))
            for angle in self.angles:
                template_a = np.array(Image.fromarray(template_s).rotate(angle, expand=True))
                pos, score = self.match_infer(img, template_a)
                positions.append(pos)
                scores.append(score)
                transformations.append(f"scale={scale},angle={angle}")
        return positions, scores, transformations

    def predict(self, img: np.ndarray) -> tuple[float, float]:
        cands = self.candidates(img)
        return cands.pos[cands.score.argmax()]

def waldo_red(img: np.ndarray):
    # Takes hsv
    return (img[:, :, 0] < 0.075) & \
           (img[:, :, 1] > 0.4) & \
           (img[:, :, 2] > 0.2)

def waldo_white(img: np.ndarray):
    # Takes hsv
    return (img[:, :, 1] < 0.1) & \
           (img[:, :, 2] > .8)

class LargestRedWhiteBlob(WaldoModel):
    kernel = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ])

    def candidates(self, img: np.ndarray) -> pd.DataFrame:
        hsv = rgb2hsv(img)
        red_blobs = dilation(closing(waldo_red(hsv), self.kernel), self.kernel)
        white_blobs = dilation(closing(waldo_white(hsv), self.kernel), self.kernel)
        united_blobs = closing(red_blobs | white_blobs, self.kernel)
        united_blobs = median_filter(united_blobs, 10)

        detected = label(united_blobs)
        blobs = pd.DataFrame(regionprops_table(detected, properties=("area", "centroid")))
        raise FjernMaxFejl
        return pd.DataFrame({
            "pos": list(zip(blobs["centroid-0"], blobs["centroid-1"])),
            "area": blobs.area,
        })

    def predict(self, img: np.ndarray) -> tuple[float, float]:
        candidates = self.candidates(img)
        return candidates.pos[candidates.area.argmax]

#def candidate_image() -> np.ndarray:

if __name__ == "__main__":
    df = pd.read_pickle("data/df.pkl")
    img = df[df.name == "14_0_3.jpg"].img.iloc[0]
    tm = TemplateMatcher(["assets/templates/eyes.jpg"])
    b = LargestRedWhiteBlob()
    c1= tm.candidates(img)#, b.candidates(img)
    breakpoint()


    # print(m.predict(img))
    #mtcnn = MTCNN(32, min_face_size=10, keep_all=True, margin=10, thresholds=[0.3, 0.4, 0.4])
    #imgs, probs = mtcnn(img, return_prob=True, save_path="face.png")
