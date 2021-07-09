import cv2 as cv
import imageio

from xray_processor import XRayProcessor

if __name__ == '__main__':
    img = "i3.png"
    img = cv.imread(img, 0)
    processed_image = XRayProcessor.unsharp_masking(img)
    filename = "i3_r.png"
    imageio.imwrite(filename, processed_image)
