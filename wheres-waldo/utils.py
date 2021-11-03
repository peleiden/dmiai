import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

vsobel3 = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
])
hsobel3 = vsobel3.T

vprewitt3 = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
])
hprewitt3 = vprewitt3.T

# Typical naming
SE1, SE2 = np.array([[0,1,0], [1, 1, 1], [0,1,0]]), np.ones((3,3))

def corr(im: np.ndarray, ker: np.ndarray):
    """ Get correlation/filter applicaiton for nxn image patch and nxn kernel/filter """
    return (im * ker).sum()

def NCC(im: np.ndarray, ker: np.ndarray):
    """ Get normalized cross correlation for nxn image patch and nxn kernel """
    return corr(im, ker) / np.sqrt(corr(im, im) * corr(ker, ker))

def lims(im: np.ndarray):
    im = im.copy()
    im[im<0] = 0
    im[im>255] = 255
    return im

def normalize(im: np.ndarray):
    return im / 255

def denormalize(im: np.ndarray):
    im = np.round(im * 255).astype(int)
    im = lims(im)
    return im.astype(int)

def fit_linear(oldmin, oldmax, newmin, newmax):
    # Calculate a and b for a linearly transformed image
    a = (newmax - newmin) / (oldmax - oldmin)
    b = newmin - a * oldmin
    return a, b

def linear(im: np.ndarray, a: float, b: float):
    im = np.round(a * im + b).astype(int)
    im = lims(im)
    return im.astype(int)

def gamma(im: np.ndarray, gamma: float):
    im = normalize(im)
    im = denormalize(im ** gamma)
    return im

def log(im: np.ndarray):
    c = 255 / np.log(1 + im.max())
    im = c * np.log(1 + im)
    return lims(np.round(im)).astype(int)

def logistic(im: np.ndarray):
    im = normalize(im.astype(float))
    im[im<0.5] = 1/2 - 1/2 * np.sqrt(1-4*im[im<0.5]**2)
    im[im>0.5] = 1/2 + 1/2 * np.sqrt(1-4*(im[im>0.5]-1)**2)
    return denormalize(im)

def prewittgradient(im: np.ndarray):
    """ The gradient of 3x3 image patch as approximated by two prewitt (3x3) filters """
    return np.sqrt(corr(im, vprewitt3)**2 + corr(im, hprewitt3)**2)

def dp(im: np.ndarray):
    """ Get accumulator and back trace image for top to bottom path
    Transpose and flip first to get other directions
    NB: Backtrace image is 1-indexed """
    acc = np.zeros_like(im, dtype=int)
    acc[0] = im[0]
    trace = np.zeros_like(im, dtype=int)
    for i in range(1, len(im)):
        for j in range(0, im.shape[1]):
            start = max(0, j-1)
            stop = min(im.shape[1]-1, j+1)
            lowest = np.argmin(acc[i-1, start:stop+1]) + start
            acc[i, j] = im[i, j] + acc[i-1, lowest]
            trace[i, j] = lowest
    # 1-indexed
    trace[1:] += 1
    return acc, trace

def min_dist_classifier(classes):
    """ See parametric_classifier for docs"""
    means = []
    for points in classes:
        means.append(np.mean(points))
    means = np.array(means)
    diffs = lambda x: np.abs(x-means)
    value_diffs = np.array([diffs(x) for x in range(256)])
    best = np.argmin(value_diffs, axis=1)
    curr = best[0]
    thresholds = {0: curr}
    # {from this point: this classifier is best}
    for i, b in enumerate(best):
        if b != curr:
            curr = b
            thresholds[i] = curr
    return thresholds

def parametric_classifier(classes):
    """ Parametric classifcation thresholds based on class data
     List of lists, each containg annotated points
     NB: This may yield more thresholds than expected as the best pdf can change when very far from annotated data
     """
    norms = []
    for points in classes:
        mu = np.mean(points)
        std = np.sqrt( np.var(points) * len(points) / (len(points)-1) )
        norms.append(norm(loc=mu, scale=std))
    probs = np.array([[n.pdf(x) for n in norms] for x in range(256)])
    best = np.argmax(probs, axis=1)
    curr = best[0]
    thresholds = {0: curr}
    # {from this point: this classifier is best}
    for i, b in enumerate(best):
        if b != curr:
            curr = b
            thresholds[i] = curr
    return thresholds

def simple_min_dist_classifier(classes):
    """ Følger ikke den andens dataformat, så de er her begge to stadig
     Note: Has been fixed to work with non-equal class sizes and non-list iters
     """
    classes = sorted(classes, key=sum)
    return [ np.mean([np.mean(cla),  np.mean(classes[j+1])]) for j, cla in enumerate(classes[:-1]) ]

def plot_parametric_classifier(classes, titles=None):
    if not titles:
        titles = range(len(classes))
    x = np.arange(np.ravel(classes).min(), np.ravel(classes).max()+1)
    for cla, title in zip(classes, titles):
        plt.plot(
            x,
            norm(loc=np.mean(cla), scale=np.std(cla, ddof=1)).pdf(x),
            label=title
        )
    plt.legend()
    plt.show()

def registration_loss(template: list, reference: list, T: np.ndarray):
    """ Computes SSE Loss between (linearly) transformed template and reference
     where each is list of points """
    F = 0
    for point, true in zip(template, reference):
        _point = T @ np.array(point)
        F += np.sum((np.array(true) - _point)**2)
    return F

def rgb2im(r: np.ndarray, g: np.ndarray, b: np.ndarray, shape=(3, 3)):
    """ Convert r, g, and b vectors in row-major order into an rgb image of shape (height, width, channels) """
    assert r.size == g.size == b.size == shape[0] * shape[1], "https://www.youtube.com/watch?v=QDRtbHd8zpc&t=3s"
    r = r.reshape(shape)
    g = g.reshape(shape)
    b = b.reshape(shape)
    return np.array([r, g, b]).transpose(1, 2, 0)


def rgb2grey(im):
    assert im.shape[2] == 3, "Gib channels bitte"
    return 0.299 * im[..., 0] + 0.5870 * im[..., 1] + 0.1140 * im[..., 2]

def rgb2si(im):
    """ Returns tuple of S, I from the HSI representation (as often; only one of these are needed) """
    assert im.shape[2] == 3, "Können sie bitte mir deiner KALEN geben?"
    S = 1 - 3*im.min(axis=2) / im.sum(axis=2)
    I = im.sum(axis=2) / 3
    return np.array((S, I))

def grey_run_length_to_im(rl, shape: tuple):
    im = np.zeros(shape[0]*shape[1], dtype=int)
    pos = 0
    for length, value in zip(rl[::2], rl[1::2]):
        im[pos:pos+length] = value
        pos += length
    return im.reshape(shape)

def binary_run_length(data):
    # Assumes zero indexed
    h = max(x[0] for x in data) + 1
    w = max(x[2] for x in data) + 1
    im = np.zeros((h, w), dtype=int)
    for dat in data:
        im[dat[0], dat[1] : dat[2]+1] = 1
    return im

def chain_to_im(start: tuple, chain):
    """ Start assumes zero-based (x, y) """
    print("WARNING: I dont fill ma boi out")
    moves = ((0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1), (1,0), (1,1))
    pos = [start[::-1]]
    for move in chain:
        move = moves[move]
        pos.append((pos[-1][0] + move[0], pos[-1][1] + move[1]))
    pos = np.transpose(pos)
    im = np.zeros((start[1]+pos[0].max(), start[0]+pos[1].max()), dtype=int)
    im[pos[0], pos[1]] = 1
    return im

def dist(v1: np.ndarray, v2: np.ndarray) -> float:
    """ The ultimate lazi boi for calculating euclidian distance between two vectors """
    diff = v1 - v2
    return np.sqrt((diff**2).sum())

def sse(v1: np.ndarray, v2: np.ndarray) -> float:
    """ Both are 2xn arrays
    Returns sse loss of the two vectors """
    diff = v1 - v2
    squared_diff = diff ** 2
    squared_sums = squared_diff.sum(axis=0)
    return squared_sums.sum()

def optimize_translation(from_: np.ndarray, to: np.ndarray):
    """ Both inputs should be 2xn arrays
    Returns a two long translation vector that minimizes mse when ADDED to from_ """
    from_mean = from_.mean(axis=1)
    to_mean = to.mean(axis=1)
    best_delta = to_mean - from_mean
    return best_delta

def minrank(im: np.ndarray) -> np.ndarray:
    """ Applies a 3x3 min rank filter """
    assert (im <= 255).all(), "minrank assumes assumptions"
    padded = 255 * np.ones((im.shape[0]+2, im.shape[1]+2), dtype=int)
    padded[1:-1, 1:-1] = im
    newim = np.zeros_like(im)
    for i in range(1, im.shape[0]+1):
        for j in range(1, im.shape[1]+1):
            newim[i-1, j-1] = padded[i-1:i+2, j-1:j+2].min()
    return newim

def medianrank(im: np.ndarray, padding=0) -> np.ndarray:
    """ Applies a 3x3 median rank filter """
    print("Warning: medianrank uses simple padding and WILL not work on edge pixels")
    padded = padding * np.ones((im.shape[0]+2, im.shape[1]+2), dtype=int)
    padded[1:-1, 1:-1] = im
    newim = np.zeros_like(im)
    for i in range(1, im.shape[0]+1):
        for j in range(1, im.shape[1]+1):
            newim[i-1, j-1] = np.median(padded[i-1:i+2, j-1:j+2])
    return newim

def maxrank(im: np.ndarray) -> np.ndarray:
    """ Applies a 3x3 max rank filter """
    padded = np.zeros((im.shape[0]+2, im.shape[1]+2), dtype=int)
    padded[1:-1, 1:-1] = im
    newim = np.zeros_like(im)
    for i in range(1, im.shape[0]+1):
        for j in range(1, im.shape[1]+1):
            newim[i-1, j-1] = padded[i-1:i+2, j-1:j+2].max()
    return newim

def com(im: np.array):
    """ Calculate center of mass of binary image
    Returned as zero-based (x, y) with origin in upper left """
    pos = np.where(im)
    ymean = np.mean(pos[0])
    xmean = np.mean(pos[1])
    return xmean, ymean

if __name__ == "__main__":
    pass
    # Example of path tracing from slides
    # im = np.array([
    #     [140, 190, 73, 19, 60],
    #     [130, 212, 14, 100, 145],
    #     [150, 20, 80, 135, 120],
    #     [157, 140, 33, 199, 100],
    #     [121, 234, 45, 210, 86],
    # ])
    # acc, trace = dp(im)
    # print(acc)
    # print(trace)

    # Example of translation optimization. Exercize 23 from 2019
    # ref = np.array([[2, 7, 1], [5, 6, 3]])
    # template = np.array([[2, 1, 8], [9, 5, 4]])
    # best_t = optimize_translation(ref, template)
    # trans = (ref.T + best_t).T
    # print(sse(trans, template))
