from __future__ import annotations
import numpy as np
from skimage.metrics import structural_similarity
from skimage.filters import gaussian
import cv2

def similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    return structural_similarity(img1, img2, multichannel=len(img1.shape)>2)

def diff(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    _, diff = structural_similarity(img1, img2, full=True, multichannel=(multichannel := len(img1.shape) > 2))
    if multichannel:
        diff = diff.mean(2)
    diff = (diff*255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_OTSU)[1]
    return thresh

def first_order_diff(seq: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return diff(seq[0], seq[1]), diff(seq[1], seq[2])

def image_preprocess(img: np.ndarray) -> np.ndarray:
    img = gaussian(img, sigma=1, multichannel=True)
    return img

if __name__ == "__main__":
    from PIL import Image
    from data import divide_rule_img
    import matplotlib.pyplot as plt

    img = np.array(Image.open("data/test18.jpeg"))
    choice_imgs = [np.array(Image.open(f"data/choices18-{i}.jpeg")) for i in range(4)]
    examples, test_imgs = divide_rule_img(img)
    examples = [[image_preprocess(e) for e in l] for l in examples]
    test_imgs = [image_preprocess(t) for t in test_imgs]

    fig, axes = plt.subplots(4, 2)
    for i in range(3):
        diffs = first_order_diff(examples[i])
        axes[i, 0].imshow(diffs[0], cmap="gray")
        axes[i, 1].imshow(diffs[1], cmap="gray")
    axes[-1, 0].imshow(d := diff(*test_imgs), cmap="gray")
    axes[-1, 1].imshow(np.ones_like(d)*255, cmap="gray")
    plt.show()
