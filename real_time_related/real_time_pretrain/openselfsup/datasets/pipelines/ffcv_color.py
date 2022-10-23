from numpy import dtype
from numpy.random import rand
import numpy as np
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from numba import njit


@njit(parallel=False, fastmath=True, inline='always')
def RGBtoGray(img):
    input_type = img.dtype
    img = img.astype(np.float32)
    gray_img = img[:, :, 0] * 0.299\
              + img[:, :, 1] * 0.587\
              + img[:, :, 2] * 0.114
    img[:,:,0] = gray_img
    img[:,:,1] = gray_img
    img[:,:,2] = gray_img
    np.clip(img, 0, 255, out=img)
    img = np.rint(img)
    img = img.astype(input_type)
    return img


class RandomGrayScale(Operation):
    """Randomly grayscale the image with probability prob.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    prob : float
        The probability with which to grayscale each image in the batch.
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        prob = self.prob

        def gray(images, dst):
            should_gray = rand(images.shape[0]) < prob
            for i in my_range(images.shape[0]):
                if should_gray[i]:
                    dst[i] = RGBtoGray(images[i])
                else:
                    dst[i] = images[i]

            return dst

        gray.is_parallel = True
        return gray

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return (previous_state, AllocationQuery(previous_state.shape, previous_state.dtype))


@njit(parallel=False, fastmath=True, inline='always')
def AdjustBrightness(image, brightness):
    input_type = image.dtype
    image = image.astype(np.float32)
    image = image * brightness
    np.clip(image, 0, 255, out=image)
    image = image.astype(input_type)
    return image


@njit(parallel=False, fastmath=True, inline='always')
def AdjustSaturation(image, saturation):
    input_type = image.dtype
    image = image.astype(np.float32)
    gry_image = RGBtoGray(image)
    blend_image = image * saturation + gry_image * (1-saturation)
    np.clip(blend_image, 0, 255, out=blend_image)
    blend_image = blend_image.astype(input_type)
    return blend_image


@njit(parallel=False, fastmath=True, inline='always')
def AdjustContrast(image, contrast):
    input_type = image.dtype
    image = image.astype(np.float32)
    mean_gray = np.mean(RGBtoGray(image))
    blend_image = image * contrast + mean_gray * (1-contrast)
    np.clip(blend_image, 0, 255, out=blend_image)
    blend_image = blend_image.astype(input_type)
    return blend_image


# From https://gist.github.com/PolarNick239/691387158ff1c41ad73c
@njit(parallel=False, fastmath=True, inline='always')
def rgb_to_hsv(rgb):
    """
    >>> from colorsys import rgb_to_hsv as rgb_to_hsv_single
    >>> 'h={:.2f} s={:.2f} v={:.2f}'.format(*rgb_to_hsv_single(50, 120, 239))
    'h=0.60 s=0.79 v=239.00'
    >>> 'h={:.2f} s={:.2f} v={:.2f}'.format(*rgb_to_hsv_single(163, 200, 130))
    'h=0.25 s=0.35 v=200.00'
    >>> np.set_printoptions(2)
    >>> rgb_to_hsv(np.array([[[50, 120, 239], [163, 200, 130]]]))
    array([[[   0.6 ,    0.79,  239.  ],
            [   0.25,    0.35,  200.  ]]])
    >>> 'h={:.2f} s={:.2f} v={:.2f}'.format(*rgb_to_hsv_single(100, 100, 100))
    'h=0.00 s=0.00 v=100.00'
    >>> rgb_to_hsv(np.array([[50, 120, 239], [100, 100, 100]]))
    array([[   0.6 ,    0.79,  239.  ],
           [   0.  ,    0.  ,  100.  ]])
    """
    input_shape = rgb.shape
    rgb = rgb.reshape(-1, 3)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc

    deltac = maxc - minc
    s = deltac / maxc
    deltac[deltac == 0] = 1  # to not divide by zero (those results in any way would be overridden in next lines)
    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac

    h = 4.0 + gc - rc
    h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
    h[r == maxc] = bc[r == maxc] - gc[r == maxc]
    h[minc == maxc] = 0.0

    h = (h / 6.0) % 1.0
    res = np.dstack((h, s, v))
    return res.reshape(input_shape)


@njit(parallel=False, fastmath=True, inline='always')
def hsv_to_rgb(hsv):
    """
    >>> from colorsys import hsv_to_rgb as hsv_to_rgb_single
    >>> 'r={:.0f} g={:.0f} b={:.0f}'.format(*hsv_to_rgb_single(0.60, 0.79, 239))
    'r=50 g=126 b=239'
    >>> 'r={:.0f} g={:.0f} b={:.0f}'.format(*hsv_to_rgb_single(0.25, 0.35, 200.0))
    'r=165 g=200 b=130'
    >>> np.set_printoptions(0)
    >>> hsv_to_rgb(np.array([[[0.60, 0.79, 239], [0.25, 0.35, 200.0]]]))
    array([[[  50.,  126.,  239.],
            [ 165.,  200.,  130.]]])
    >>> 'r={:.0f} g={:.0f} b={:.0f}'.format(*hsv_to_rgb_single(0.60, 0.0, 239))
    'r=239 g=239 b=239'
    >>> hsv_to_rgb(np.array([[0.60, 0.79, 239], [0.60, 0.0, 239]]))
    array([[  50.,  126.,  239.],
           [ 239.,  239.,  239.]])
    """
    input_shape = hsv.shape
    hsv = hsv.reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    i = (h * 6.0).astype(np.int32)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    rgb = np.zeros_like(hsv)
    v = np.expand_dims(v, 1)
    t = np.expand_dims(t, 1)
    p = np.expand_dims(p, 1)
    q = np.expand_dims(q, 1)
    rgb[i == 0] = np.hstack((v, t, p))[i == 0]
    rgb[i == 1] = np.hstack((q, v, p))[i == 1]
    rgb[i == 2] = np.hstack((p, v, t))[i == 2]
    rgb[i == 3] = np.hstack((p, q, v))[i == 3]
    rgb[i == 4] = np.hstack((t, p, v))[i == 4]
    rgb[i == 5] = np.hstack((v, p, q))[i == 5]
    rgb[s == 0.0] = np.hstack((v, v, v))[s == 0.0]

    return rgb.reshape(input_shape)


@njit(parallel=False, fastmath=True, inline='always')
def AdjustHue(image, hue):
    input_type = image.dtype
    image = image.astype(np.float32)
    hsv = rgb_to_hsv(image)
    np_h = hsv[:,:,0]
    np_h = (np_h * 255).astype(np.uint8)
    np_h = np_h.astype(np.float32)
    np_h = (np_h + hue*255) % 255.0
    np_h = np_h / 255.0
    hsv[:,:,0] = np_h
    rgb = hsv_to_rgb(hsv)
    np.clip(rgb, 0, 255, out=rgb)
    rgb = rgb.astype(input_type)
    return rgb


class RandomColorJitter(Operation):
    """Randomly color-jitter the image with probability prob.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    prob : float
        The probability with which to color-jitter each image in the batch.
    brightness, contrast, saturation, hue: float
        The same as torchvision.transforms.ColorJitter
    """
    def __init__(
            self, prob: float = 0.5,
            brightness = 0,
            contrast = 0,
            saturation = 0,
            hue = 0,
            ):
        super().__init__()
        self.prob = prob
        self.brightness = (max(0, 1-brightness), 1+brightness)
        self.contrast = (max(0, 1-contrast), 1+contrast)
        self.saturation = (max(0, 1-saturation), 1+saturation)
        self.hue = (-hue, hue)

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        prob = self.prob
        brightness = self.brightness
        contrast = self.contrast
        saturation = self.saturation
        hue = self.hue

        def color_jitter(images, dst):
            should_jitter = rand(images.shape[0]) < prob
            all_orders = rand(images.shape[0] * 4)
            all_brightness = rand(images.shape[0]) * (brightness[1] - brightness[0]) + brightness[0]
            all_contrast = rand(images.shape[0]) * (contrast[1] - contrast[0]) + contrast[0]
            all_saturation = rand(images.shape[0]) * (saturation[1] - saturation[0]) + saturation[0]
            all_hue = rand(images.shape[0]) * (hue[1] - hue[0]) + hue[0]
            for i in my_range(images.shape[0]):
                if should_jitter[i]:
                    dst[i] = images[i]
                    curr_orders = np.argsort(all_orders[i*4 : (i+1)*4])
                    for _order in curr_orders:
                        if _order == 0:
                            dst[i] = AdjustBrightness(dst[i], all_brightness[i])
                        elif _order == 1:
                            dst[i] = AdjustContrast(dst[i], all_contrast[i])
                        elif _order == 2:
                            dst[i] = AdjustSaturation(dst[i], all_saturation[i])
                        else:
                            dst[i] = AdjustHue(dst[i], all_hue[i])
                else:
                    dst[i] = images[i]
            return dst

        color_jitter.is_parallel = True
        return color_jitter

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return (previous_state, AllocationQuery(previous_state.shape, previous_state.dtype))


class RandomColorJitterNoHue(RandomColorJitter):
    """Randomly color-jitter the image with probability prob.
    Operates on raw arrays (not tensors).
    No hue augmentation.

    Parameters
    ----------
    prob : float
        The probability with which to color-jitter each image in the batch.
    brightness, contrast, saturation, hue: float
        The same as torchvision.transforms.ColorJitter
    """
    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        prob = self.prob
        brightness = self.brightness
        contrast = self.contrast
        saturation = self.saturation

        def color_jitter(images, dst):
            should_jitter = rand(images.shape[0]) < prob
            all_orders = rand(images.shape[0] * 3)
            all_brightness = rand(images.shape[0]) * (brightness[1] - brightness[0]) + brightness[0]
            all_contrast = rand(images.shape[0]) * (contrast[1] - contrast[0]) + contrast[0]
            all_saturation = rand(images.shape[0]) * (saturation[1] - saturation[0]) + saturation[0]
            for i in my_range(images.shape[0]):
                if should_jitter[i]:
                    dst[i] = images[i]
                    curr_orders = np.argsort(all_orders[i*3 : (i+1)*3])
                    for _order in curr_orders:
                        if _order == 0:
                            dst[i] = AdjustBrightness(dst[i], all_brightness[i])
                        elif _order == 1:
                            dst[i] = AdjustContrast(dst[i], all_contrast[i])
                        else:
                            dst[i] = AdjustSaturation(dst[i], all_saturation[i])
                else:
                    dst[i] = images[i]
            return dst

        color_jitter.is_parallel = True
        return color_jitter
