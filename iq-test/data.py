from __future__ import annotations
import numpy as np

S = 110

def divide_rule_img(img: np.ndarray) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
    examples = list()
    for i in range(3):
        examples.append(
            [
                img[i*S:(i+1)*S, 0:S],
                img[i*S:(i+1)*S, 2*S:3*S],
                img[i*S:(i+1)*S, 4*S:5*S],
            ]
        )
    return examples, [img[3*S:, 0:S], img[3*S:, 2*S:3*S]]

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    img = np.array(Image.open("data/test18.jpeg"))
    examples, condition = divide_rule_img(img)
    for imgs in examples:
        fig, axes = plt.subplots(1, 3)
        for i, img in enumerate(imgs):
            axes[i].imshow(img)
        plt.show()

    fig, axes = plt.subplots(1,2)
    for i, img in enumerate(condition):
        axes[i].imshow(img)
    plt.show()
