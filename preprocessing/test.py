import imageio

from xray_processor import XRayProcessor

if __name__ == '__main__':
    img = "img.jpg"
    processed_image = XRayProcessor.clahe(img)
    filename = "result_hef.jpg"
    imageio.imwrite(filename, processed_image)
