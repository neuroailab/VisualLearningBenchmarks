from numpy.random import rand
import numpy as np
import pdb
from PIL import Image
import torchvision.transforms.functional as torch_func

import openselfsup.datasets.pipelines.ffcv_color as ffcv_color


def get_random_image():
    np.random.seed(0)
    input_img = rand(224, 224, 3) * 255
    input_img = input_img.astype(np.uint8)
    input_img_PIL = Image.fromarray(input_img)
    return input_img, input_img_PIL

def check_diff(input_img, ffcv_output, torch_output):
    diff = np.abs(ffcv_output.astype(np.float32) - torch_output.astype(np.float32))
    print('Image size {}, Different pixels {}, maximal differences {}'.format(
        224, np.sum(diff > 0), np.max(diff)))
    diff_flag = np.sum(diff, axis=-1) > 0
    print('Input image: ', input_img[diff_flag][:12])
    print('FFCV output: ', ffcv_output[diff_flag][:12])
    print('Torch output: ', torch_output[diff_flag][:12])

def test_RGBtoGray():
    input_img, input_img_PIL = get_random_image()

    ffcv_output = ffcv_color.RGBtoGray(input_img)
    torch_output = torch_func.rgb_to_grayscale(input_img_PIL, 3)
    torch_output = np.asarray(torch_output)
    check_diff(input_img, ffcv_output, torch_output)

def test_brightness():
    input_img, input_img_PIL = get_random_image()

    ffcv_output = ffcv_color.AdjustBrightness(input_img, 1.2)
    torch_output = np.asarray(torch_func.adjust_brightness(input_img_PIL, 1.2))
    check_diff(input_img, ffcv_output, torch_output)

def test_saturation():
    input_img, input_img_PIL = get_random_image()

    ffcv_output = ffcv_color.AdjustSaturation(input_img, 1.2)
    torch_output = np.asarray(torch_func.adjust_saturation(input_img_PIL, 1.2))
    check_diff(input_img, ffcv_output, torch_output)

def test_contrast():
    input_img, input_img_PIL = get_random_image()

    ffcv_output = ffcv_color.AdjustContrast(input_img, 1.2)
    torch_output = np.asarray(torch_func.adjust_contrast(input_img_PIL, 1.2))
    check_diff(input_img, ffcv_output, torch_output)

def test_hue():
    input_img, input_img_PIL = get_random_image()

    ffcv_output = ffcv_color.AdjustHue(input_img, 0.2)
    torch_output = np.asarray(torch_func.adjust_hue(input_img_PIL, 0.2))
    check_diff(input_img, ffcv_output, torch_output)

def test_rgb_to_hsv():
    input_img, input_img_PIL = get_random_image()
    ffcv_output = ffcv_color.rgb_to_hsv(input_img)
    ffcv_output_h = (ffcv_output[:, :, 0] * 255).astype(np.uint8)
    torch_output = np.asarray(input_img_PIL.convert("HSV"))
    torch_output_h = torch_output[:, :, 0]
    print(np.sum(ffcv_output_h != torch_output_h))
    pdb.set_trace()
    pass

def test_hsv_to_rgb():
    input_img, input_img_PIL = get_random_image()
    hsv_from_ffcv = ffcv_color.rgb_to_hsv(input_img)
    print(np.allclose(
        input_img, 
        ffcv_color.hsv_to_rgb(hsv_from_ffcv)))
    hsv_output = np.asarray(input_img_PIL.convert("HSV"))
    hsv_output = hsv_output.astype(np.float32)
    hsv_output[:, :, 0] /= 255
    hsv_output[:, :, 1] /= 255
    rgb_from_PIL_hsv = np.rint(ffcv_color.hsv_to_rgb(hsv_output)).astype(np.uint8)
    diff_flag = np.sum(np.abs(input_img - rgb_from_PIL_hsv), axis=-1) > 0
    print(np.allclose(input_img, rgb_from_PIL_hsv))
    print('Input Image: ', input_img[diff_flag][:5])
    print('HSV from FFCV: ', hsv_from_ffcv[diff_flag][:5])
    hsv_output = np.asarray(input_img_PIL.convert("HSV"))
    print('HSV from PIL: ', hsv_output[diff_flag][:5])
    print('New RGB Image: ', rgb_from_PIL_hsv[diff_flag][:5])
    pdb.set_trace()


if __name__ == '__main__':
    #test_RGBtoGray()
    #test_brightness()
    #test_saturation()
    #test_contrast()
    #test_hue()
    #test_hsv()
    test_hsv_to_rgb()
