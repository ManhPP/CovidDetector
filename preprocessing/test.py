import imageio

from xray_processor import XRayProcessor

if __name__ == '__main__':
    img = "img.jpg"
    processed_image = XRayProcessor.unsharp_masking(img)
    filename = "result_2.jpg"
    imageio.imwrite(filename, processed_image)
