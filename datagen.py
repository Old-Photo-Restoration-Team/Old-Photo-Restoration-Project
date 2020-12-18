# Python modules
from random import randrange, choice, uniform
import logging
import os

# Pip modules
import cv2
import torchvision
import numpy as np
from tqdm import tqdm


def apply_gaussian_noise(image):
    """Adds gausian noise to an image and returns the noisy image

    Args:
        image (array-like): image to alter

    Returns:
        array-like: noisy image
    """

    h, w, c = image.shape
    mu = 0

    # As per the article - select sigma in [5, 50]
    # Check randrange or uniform for article
    sigma = randrange(5, 50)
    noise = np.random.normal(mu, sigma, (h, w, c))

    noisy_image = image + noise

    noisy_img_clipped = noisy_image.clip(min=0, max=255)

    return noisy_img_clipped.astype("uint8")


def apply_gaussian_blur(image):
    """Applies Gaussian blur to image using randomly generated kernel size and sigma value

    Args:
        image (array-like): image to alter

    Returns:
        array-like: blurred image
    """
    kval = choice((3, 5, 7))
    ksize = (kval, kval)
    sigma = uniform(1.0, 5.0)

    blurred_image = cv2.GaussianBlur(image, ksize, sigma)

    return blurred_image


def apply_jpeg_compression(image):
    """Compresses the input image with JPEG

    Args:
        image (array-like): image to alter

    Returns:
        array-like: altered image after compression/decompression
    """
    quality = randrange(40, 100)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encoded_img = cv2.imencode(".jpg", image, encode_param)
    if result == False:
        logging.warning("Image could not be compressed with JPEG")
    decoded_img = cv2.imdecode(encoded_img, 1)

    return decoded_img


def apply_RGB_shift(image):
    """Applies a pixel level value shift on RGB to the image

    *** Can create some artifacts ***

    Args:
        image (array-like): image to alter

    Returns:
        array-like: color-jitter image
    """

    image = image.astype("float32")

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    range_ = 20

    R_shift = randrange(-1 * range_, range_)
    G_shift = randrange(-1 * range_, range_)
    B_shift = randrange(-1 * range_, range_)

    shifted_img = np.dstack((R + R_shift, G + G_shift, B + B_shift))

    shifted_img_clipped = shifted_img.clip(0, 255)

    return shifted_img_clipped.astype("uint8")


def apply_box_blur(image):
    """Applies box blur to image

    Args:
        image (array-like): image to alter

    Returns:
        array-like: blurred image
    """
    blurred_image = cv2.blur(image, (3, 3))
    return blurred_image


def apply_all_degradations(image):
    """Applies sequentially all degradations to the input image

    Args:
        image (array-like): image to alter

    Returns:
        array-like: altered image
    """
    res_im = image
    res_im = apply_gaussian_noise(res_im)
    res_im = apply_gaussian_blur(res_im)
    res_im = apply_jpeg_compression(res_im)
    res_im = apply_RGB_shift(res_im)
    res_im = apply_box_blur(res_im)

    return res_im
