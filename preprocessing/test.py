import imageio

from xray_processor import XRayProcessor

if __name__ == '__main__':
    img = "img_.jpg"
    processed_image = XRayProcessor.combine_preprocessing(img)
    filename = "result.jpg"
    imageio.imwrite(filename, processed_image)
