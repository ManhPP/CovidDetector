import copy

import cv2 as cv
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
from skimage.filters import unsharp_mask

import preprocessing.utils as utils


class XRayProcessor:

    @staticmethod
    def beasf(image, gamma=0.5):
        # image = cv.imread(filename, 0)
        m = int(np.mean(image, dtype=np.int32))
        h = np.histogram(image, bins=256)[0] / (image.shape[0] * image.shape[1])
        h_lower = utils.sub_hist(image_pdf=h, minimum=0, maximum=m, normalize=True)
        h_upper = utils.sub_hist(image_pdf=h, minimum=m, maximum=255, normalize=True)

        cdf_lower = utils.CDF(hist=h_lower)
        cdf_upper = utils.CDF(hist=h_upper)

        # Find x | CDF(x) = 0.5
        half_low = 0
        for idx in range(0, m + 2):
            if cdf_lower[idx] > 0.5:
                half_low = idx
                break
        half_up = 0
        for idx in range(m, 256):
            if cdf_upper[idx + 1] > 0.5:
                half_up = idx
                break

        # sigmoid CDF creation
        tones_low = np.arange(0, m + 1, 1)
        x_low = 5.0 * (tones_low - half_low) / m  # shift & scale intensity x to place sigmoid [-2.5, 2.5]
        s_low = 1 / (1 + np.exp(-gamma * x_low))  # lower sigmoid

        tones_up = np.arange(m, 256, 1)
        x_up = 5.0 * (tones_up - half_up) / (255 - m)  # shift & scale intensity x to place sigmoid [-2.5, 2.5]
        s_up = 1 / (1 + np.exp(-gamma * x_up))  # upper sigmoid

        mapping_vector = np.zeros(shape=(256,))
        for idx in range(0, m + 1):
            mapping_vector[idx] = np.int32(m * s_low[idx])

        minimum = mapping_vector[0]
        maximum = mapping_vector[m]
        for idx in range(0, m + 1):
            mapping_vector[idx] = np.int32((m / (maximum - minimum)) * (mapping_vector[idx] - minimum))
        for idx in range(m + 1, 256):
            mapping_vector[idx] = np.int32(m + (255 - m) * s_up[idx - m - 1])

        minimum = mapping_vector[m + 1]
        maximum = mapping_vector[255]
        for idx in range(m + 1, 256):
            mapping_vector[idx] = (255 - m) * (mapping_vector[idx] - minimum) / (maximum - minimum) + m

        res = copy.deepcopy(image)
        res[:, :] = mapping_vector[image[:, :]]
        # res = cv.cvtColor(res, cv.COLOR_GRAY2RGB)
        return res

    @staticmethod
    def clahe(img, clip_limit=4, window_size=8):
        # img = cv.imread(filename, 0)
        # create a CLAHE object (Arguments are optional).
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(window_size, window_size))
        cl1 = clahe.apply(img)
        # cl1 = cv.cvtColor(cl1, cv.COLOR_GRAY2RGB)
        return cl1

    @staticmethod
    def hef(img_grayscale, d0v=1):
        """Runs the algorithm for the image."""
        assert 1 <= d0v <= 90
        # img_grayscale = cv.imread(filename,  0)

        img = utils.normalize(np.min(img_grayscale), np.max(img_grayscale), 0, 255,
                              img_grayscale)
        # HF part
        img_fft = fft2(img)  # img after fourier transformation
        img_sfft = fftshift(img_fft)  # img after shifting component to the center

        m, n = img_sfft.shape
        filter_array = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                filter_array[i, j] = 1.0 - np.exp(- ((i - m / 2.0) ** 2 + (j - n / 2.0) ** 2) / (2 * (d0v ** 2)))
        k1 = 0.5
        k2 = 0.75
        high_filter = k1 + k2 * filter_array

        img_filtered = high_filter * img_sfft
        img_hef = np.real(ifft2(fftshift(img_filtered)))  # HFE filtering done

        # HE part
        # Building the histogram
        hist, bins = utils.histogram(img_hef)
        # Calculating probability for each pixel
        pixel_probability = hist / hist.sum()
        # Calculating the CDF (Cumulative Distribution Function)
        cdf = np.cumsum(pixel_probability)
        cdf_normalized = cdf * 255
        hist_eq = {}
        for i in range(len(cdf)):
            hist_eq[bins[i]] = int(cdf_normalized[i])

        for i in range(m):
            for j in range(n):
                img_grayscale[i][j] = hist_eq[img_hef[i][j]]

        # img_grayscale = cv.cvtColor(img_grayscale, cv.COLOR_GRAY2RGB)

        return img_grayscale

    @staticmethod
    def unsharp_masking(image, radius=5, amount=2):
        # image = imageio.imread(filename)
        # image = img_as_float(image)
        #
        # if filter_type == "gauss":
        #     blurred_image = gaussian_filter(image, sigma=radius)
        # elif filter_type == "median":
        #     blurred_image = median_filter(image, size=20)
        # else:
        #     assert "not support"
        #
        # mask = image - blurred_image
        # sharpened_image = image + mask * amount
        #
        # sharpened_image = np.clip(sharpened_image, float(0), float(1))
        # sharpened_image = (sharpened_image * 255).astype(np.uint8)

        sharpened_image = unsharp_mask(image, radius, amount)
        sharpened_image = (sharpened_image * 255).astype(np.uint8)
        return sharpened_image

    @staticmethod
    def combine_preprocessing(filename, radius=3, amount=1, clip_limit=4, window_size=8, gamma=0.5, d0v=1):
        image = cv.imread(filename, 0)
        image = XRayProcessor.unsharp_masking(image, radius, amount)

        image = XRayProcessor.clahe(image, clip_limit, window_size)
        image = XRayProcessor.beasf(image, gamma)

        # image = XRayProcessor.hef(image, d0v)

        image = cv.medianBlur(image, 5)

        img_grayscale = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

        return img_grayscale


