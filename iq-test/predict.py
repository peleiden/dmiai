from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from imgutils import image_preprocess
import model


@dataclass
class Case:
    examples: list[list[np.ndarray]]
    problem: list[np.ndarray]

    def __post_init__(self):
        for i in range(len(self.examples)):
            for j in range(len(self.examples[i])):
                self.examples[i][j] = image_preprocess(self.examples[i][j])
        for i in range(len(self.problem)):
            self.problem[i] = image_preprocess(self.problem[i])

    def __hash__(self) -> int:
        return round(sum(x.sum() for ex in self.examples for x in ex))

def crop_to_size(im: np.ndarray) -> np.ndarray:
    assert im.shape[0] == im.shape[1]
    extra_size = im.shape[0] - 110
    assert extra_size >= 0
    dt = extra_size // 2
    return im[dt:dt+110, dt:dt+110].copy()

transforms = model.ConsistentTransformations()
actions = list(transforms.transformation_library.values())

# Consider a transformation to fit, if the average pixel change is less than 5 (in 255 domain)
moe = 15
def im_dist(im1: np.ndarray, im2: np.ndarray) -> float:
    # Takes images in [0, 1] domain and return avg. difference in 255 domain
    return np.abs(im1-im2).mean() * 255

def _predict_by_fo_diff(case: Case, options: list[np.ndarray]) -> int | None:
    lim = 0.05
    diff_pixels_ratios = np.zeros(3)
    for i, example in enumerate(case.examples):
        d1 = np.abs(example[1]-example[0])
        d2 = np.abs(example[2]-example[1])
        try:
            diff_pixels_ratios[i] = (d1>lim).sum() / (d2>lim).sum()
        except ZeroDivisionError:
            diff_pixels_ratios[i] = 0
    if np.all((diff_pixels_ratios > 0.2) & (diff_pixels_ratios < 5)):
        # Assume we can somehow use this wacky algorithm to guess it
        rats = np.zeros(4)
        d = np.abs(case.problem[1]-case.problem[0])
        for i, opt in enumerate(options):
            di = np.abs(opt-case.problem[1])
            try:
                rats[i] = (d>lim).sum() / (di>lim).sum()
            except ZeroDivisionError:
                rats[i] = np.inf
        return int(np.argmin(np.abs(rats-diff_pixels_ratios.mean())))

def predict(case: Case, options: list[np.ndarray]) -> int:
    global actions
    """ Returns the index of the option that fits the best
    Assumes all images in case and options have been exposed to same
    preprocessing. All images should also be in [0, 1] domain """
    dists = np.zeros((len(actions), 3, 2))

    for i, action in enumerate(actions):
        for j, example in enumerate(case.examples):
            current_im = example[0].copy()
            for k in range(1, 3):
                current_im = crop_to_size(action(current_im))
                dists[i, j, k-1] = im_dist(current_im, example[k])
    # We now know the approximate distance to the true pictures under all transformations
    # Now choose what fits best
    # Array of action indices that are candidates for the final model
    last_within_moe = np.where((dists[..., 1]<moe).all(axis=1))[0]
    both_within_moe = np.where((dists[..., 0]<moe).all(axis=1) & (dists[..., 1]<moe).all(axis=1))[0]

    if both_within_moe.size == 1:
        # One action is much better than all others, so choose that
        action = actions[both_within_moe[0]]
    elif both_within_moe.size:
        # Multiple actions are within moe, so choose the one with lowest avg. dist
        action = actions[both_within_moe[np.argmin([dists[idx].mean() for idx in  both_within_moe])]]
    elif last_within_moe.size == 1:
        # One action is much better than all others, so choose that
        action = actions[last_within_moe[0]]
    elif last_within_moe.size:
        action = actions[last_within_moe[np.argmin([dists[idx].mean() for idx in  last_within_moe])]]
    else:
        # At this point, none of our transformations work, so we look at first order differences
        fo_pred = _predict_by_fo_diff(case, options)

        if fo_pred is None:
            # Truly out of ideas, so do random guess
            fo_pred = 0
        return fo_pred

    opt_dists = list()
    pred_image = crop_to_size(action(crop_to_size(action(case.problem[0]))))
    for opt_im in options:
        opt_dists.append(im_dist(pred_image, opt_im))

    return int(np.argmin(opt_dists))
