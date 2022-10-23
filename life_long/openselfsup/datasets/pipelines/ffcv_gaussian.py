from numpy import dtype
from numpy.random import rand
import numpy as np
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from numba import njit, prange
import pdb


@njit(parallel=True, fastmath=True, inline='always')
def ImagingLineBoxBlur(
        lineIn, lastx,
        radius, edgeA, edgeB,
        ww, fw):
    lineOut = np.zeros_like(lineIn)
    acc = 0
    bulk = 0

    def MOVE_ACC(acc, subtract, add):
        acc += lineIn[add] - lineIn[subtract]
        return acc

    def ADD_FAR(bulk, acc, left, right):
        bulk = (acc * ww) + (lineIn[left] + lineIn[right]) * fw
        return bulk

    def SAVE(x, bulk):
        lineOut[x] = int(bulk + (1 << 23)) >> 24

    acc = lineIn[0] * (radius + 1)
    for x in prange(edgeA-1):
        acc += lineIn[x]
    acc += lineIn[lastx] * (radius - edgeA + 1)

    if edgeA > edgeB:
        _tmp = edgeA
        edgeA = edgeB
        edgeB = _tmp

    for x in prange(edgeA):
        acc = MOVE_ACC(acc, 0, x + radius)
        bulk = ADD_FAR(bulk, acc, 0, x + radius + 1)
        SAVE(x, bulk)

    for x in prange(edgeA, edgeB):
        acc = MOVE_ACC(acc, x - radius - 1, x + radius)
        bulk = ADD_FAR(bulk, acc, x - radius - 1, x + radius + 1)
        SAVE(x, bulk)

    for x in prange(edgeB, lastx+1):
        acc = MOVE_ACC(acc, x - radius - 1, lastx)
        bulk = ADD_FAR(bulk, acc, x - radius - 1, lastx)
        SAVE(x, bulk)
    return lineOut


@njit(parallel=True, fastmath=True, inline='always')
def ImagingHorizontalBoxBlur(img, floatRadius):
    radius = int(floatRadius)
    ww = int((1 << 24) / (floatRadius * 2 + 1))
    fw = ((1 << 24) - (radius * 2 + 1) * ww) / 2

    edgeA = min(radius + 1, img.shape[0])
    edgeB = max(img.shape[0] - radius - 1, 0)

    for yc in prange(img.shape[1] * img.shape[2]):
        y = yc // img.shape[2]
        c = yc % img.shape[2]
        img[:, y, c] = ImagingLineBoxBlur(
                img[:, y, c], 
                img.shape[0] - 1,
                radius, edgeA, edgeB,
                ww, fw)
    return img


@njit(parallel=True, fastmath=True, inline='always')
def ImagingVerticalBoxBlur(img, floatRadius):
    radius = int(floatRadius)
    ww = int((1 << 24) / (floatRadius * 2 + 1))
    fw = ((1 << 24) - (radius * 2 + 1) * ww) / 2

    edgeA = min(radius + 1, img.shape[1])
    edgeB = max(img.shape[1] - radius - 1, 0)

    for xc in prange(img.shape[0] * img.shape[2]):
        x = xc // img.shape[2]
        c = xc % img.shape[2]
        img[x, :, c] = ImagingLineBoxBlur(
                img[x, :, c], 
                img.shape[1] - 1,
                radius, edgeA, edgeB,
                ww, fw)
    return img


@njit(parallel=False, fastmath=True, inline='always')
def ImagingBoxBlur(img, radius, n):
    img = img.astype(np.uint32)
    for _ in range(n):
        img = ImagingHorizontalBoxBlur(img, radius)
    for _ in range(n):
        img = ImagingVerticalBoxBlur(img, radius)
    img = img.astype(np.uint8)
    return img


@njit(parallel=False, fastmath=True, inline='always')
def GaussianBlur(img, radius):
    passes = 3
    sigma2 = radius * radius / passes
    L = np.sqrt(12.0 * sigma2 + 1.0)
    l = np.floor((L - 1.0) / 2.0)
    a = (2 * l + 1) * (l * (l + 1) - 3 * sigma2)
    a /= 6 * (sigma2 - (l + 1) * (l + 1))
    return ImagingBoxBlur(img, l + a, passes)


class RandomGaussianBlur(Operation):
    """Randomly grayscale the image with probability prob.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    prob : float
        The probability with which to grayscale each image in the batch.
    """

    def __init__(
            self, prob: float = 0.5, 
            sigma_min: float=0.1,
            sigma_max: float=2.0,
            ):
        super().__init__()
        self.prob = prob
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        prob = self.prob
        sigma_max = self.sigma_max
        sigma_min = self.sigma_min

        def gaussian_blur(images, dst):
            should_gaussian = rand(images.shape[0]) < prob
            gaussian_sigmas = \
                    rand(images.shape[0]) * (sigma_max - sigma_min) \
                    + sigma_min
            for i in my_range(images.shape[0]):
                if should_gaussian[i]:
                    _gaussian_sigma = gaussian_sigmas[i]
                    dst[i] = GaussianBlur(images[i], _gaussian_sigma)
                else:
                    dst[i] = images[i]
            return dst

        gaussian_blur.is_parallel = True
        return gaussian_blur

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return (previous_state, AllocationQuery(previous_state.shape, previous_state.dtype))
