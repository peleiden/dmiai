from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import chain
from typing import Callable
from functools import partial

import numpy as np
import pandas as pd
from skimage.color.colorconv import rgb2hsv
from PIL import Image

from data import divide_rule_img
from imgutils import first_order_diff, image_preprocess, diff, similarity

class IQModel(ABC):
    def receive(self, rule_img: np.ndarray, choice_imgs: list[np.ndarray]) -> tuple[list[list[np.ndarray]], list[np.ndarray], list[np.ndarray]]:
        examples, test_imgs = divide_rule_img(rule_img)
        examples = [[image_preprocess(e) for e in l] for l in examples]
        test_imgs = [image_preprocess(t) for t in test_imgs]
        choice_imgs = [image_preprocess(c) for c in choice_imgs]
        return examples, test_imgs, choice_imgs

    @abstractmethod
    def predict(self, rule_img: np.ndarray, choice_imgs: list[np.ndarray]) -> int:
        pass

class BaselineModel(IQModel):
    def predict(self, rule_img: np.ndarray, choice_imgs: list[np.ndarray]) -> int:
        return 0 # Chosen very randomly

class UniformDifferences(IQModel):
    def __init__(self):
        self.scorer = np.median

    def predict(self, rule_img: np.ndarray, choice_imgs: list[np.ndarray]) -> int:
        examples, test_imgs, choice_imgs = self.receive(rule_img, choice_imgs)
        diff_pairs = [first_order_diff(ex) for ex in examples]
        test_diff = diff(test_imgs[0], test_imgs[1])
        all_diffs = list(chain(*diff_pairs)) + [test_diff]

        candidate_diffs = [diff(test_imgs[1], c) for c in choice_imgs]

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4)
        for i, d in enumerate(candidate_diffs):
            axes[i].imshow(d, cmap="gray")
        plt.show()

        scores = list()
        for c in candidate_diffs:
            similis = [similarity(c, known_diff) for known_diff in all_diffs]
            scores.append(self.scorer(similis))
        return np.argmax(scores)

class HighestSimilarity(IQModel):
    bins = 10

    def hist(self, img: np.ndarray):
        return np.histogram(img, bins=self.bins, range=(0, 1))[0]

    def predict(self, rule_img: np.ndarray, choice_imgs: list[np.ndarray]) -> int:
        examples, test_imgs, choice_imgs = self.receive(rule_img, choice_imgs)
        test_imgs = [rgb2hsv(t) for t in test_imgs]
        choice_imgs = [rgb2hsv(c) for c in choice_imgs]
        test_dists = [self.hist(t[:, :, 0]) for t in test_imgs]
        choice_dists = [self.hist(c[:, :, 0]) for c in choice_imgs]

        options = dict(choice=list(), test=list(), mse=list())
        for i, c in enumerate(choice_dists):
            for j, t in enumerate(test_dists):
                options["choice"].append(i)
                options["test"].append(j)
                options["mse"].append(
                    ((c - t)**2).mean()
                )
        options = pd.DataFrame(options)
        mean_mses = options.groupby("choice").mean()
        return mean_mses.mse.argmin()

class ConsistentTransformations(IQModel):
    def __init__(self):
        self.consistency_threshold = 1.10
        self.angles = np.arange(1, 8) * 45
        self.transformation_library = dict()
        for a in self.angles:
            self.transformation_library[f"rot{a}"] = partial(self.rotation, a)

    def rotation(self, angle:float, img: np.ndarray) -> np.ndarray:
        return (np.array(
            Image.fromarray(
                (255*img).astype(np.uint8)
            ).rotate(angle, expand=True, fillcolor=(255, 255, 255))
        )/255).astype(np.float64)

    def pad_for_similarity(self, img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = img1.shape[0] - img2.shape[0]
        if d:
            margins = np.array([(int(np.ceil(d / 2)), int(np.floor(d / 2)))]*2+[(0,0)])
            if d < 0:
                img1 = np.pad(img1, pad_width=margins, constant_values=1)
            else:
                img2 = np.pad(img2, pad_width=margins, constant_values=1)
        return img1, img2

    def consistency(self, imgs: np.ndarray, transform: Callable) -> float:
        sims = list()
        for i, img in enumerate(imgs[:-1]):
            sims.append(
                similarity(*self.pad_for_similarity(transform(img), imgs[i+1]))
                /
                similarity(img, imgs[i+1])
            )
        return np.mean(sims)

    def predict(self, rule_img: np.ndarray, choice_imgs: list[np.ndarray]) -> int:
        examples, test_imgs, choice_imgs = self.receive(rule_img, choice_imgs)
        for name, fun in self.transformation_library.items():
            example_consistensies = [self.consistency(example, fun) for example in examples]
            print(name, example_consistensies)
            if sum(c > self.consistency_threshold for c in example_consistensies) > 2:
                choice_consistensies = [self.consistency([*test_imgs, c], fun) for c in choice_imgs]
                print(choice_consistensies)
                if any(c > self.consistency_threshold for c in choice_consistensies):
                    print(name)
                    return np.argmax(choice_consistensies)
        return None

if __name__ == "__main__":
    from data import divide_rule_img

    img = np.array(Image.open("data/test18.jpeg"))
    choice_imgs = [np.array(Image.open(f"data/choices18-{i}.jpeg")) for i in range(4)]
    #print(UniformDifferences().predict(img, choice_imgs))
    #print(HighestSimilarity().predict(img, choice_imgs))
    print(ConsistentTransformations().predict(img, choice_imgs))
