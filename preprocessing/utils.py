from collections import OrderedDict

import numpy as np


def normalize(min_old, max_old, min_new, max_new, val):
    """
    Normalizes values to the interval [min_new, max_new]
	:param	min_old: min value from old base.
	:param	max_old: max value from old base.
	:param	min_new: min value from new base.
	:param	max_new: max value from new base.
	:param	val: float or array-like value to be normalized.
	"""

    ratio = (val - min_old) / (max_old - min_old)
    normalized = (max_new - min_new) * ratio + min_new
    return normalized.astype(np.uint8)


def histogram(data):
    """
    Generates the histogram for the given data.
	:param	data: data to make the histogram.
	:return: histogram, bins
    """
    pixels, count = np.unique(data, return_counts=True)
    hist = OrderedDict()

    for i in range(len(pixels)):
        hist[pixels[i]] = count[i]

    return np.array(list(hist.values())), np.array(list(hist.keys()))


def to_grayscale(image):
    red_v = image[:, :, 0] * 0.299
    green_v = image[:, :, 1] * 0.587
    blue_v = image[:, :, 2] * 0.144
    image = red_v + green_v + blue_v

    return image.astype(np.uint8)


def sub_hist(image_pdf, minimum, maximum, normalize):
    """
    Compute the subhistogram between [minimum, maximum] of a given histogram image_pdf
    :param image_pdf: numpy.array
    :param minimum: int
    :param maximum: int
    :param normalize: boolean
    :return: numpy.array
    """
    hi = np.zeros(shape=image_pdf.shape)
    total = 0
    for idx in range(minimum, maximum + 1):
        total += image_pdf[idx]
        hi[idx] = image_pdf[idx]
    if normalize:
        for idx in range(minimum, maximum + 1):
            hi[idx] /= total
    return hi


def CDF(hist):
    """
    Compute the CDF of the input histogram
    :param hist: numpy.array()
    :return: numpy.array()
    """
    cdf = np.zeros(shape=hist.shape)
    cdf[0] = hist[0]
    for idx in range(1, len(hist)):
        cdf[idx] = cdf[idx - 1] + hist[idx]
    return cdf
