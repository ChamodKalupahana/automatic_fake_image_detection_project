import numpy as np
import matplotlib.pyplot as plt

def load_image(image):
    """ read the string input and returns the full image 

    Args:
        image (str): name of image

    Returns:
        orginal_image
    """
    # read the image and divude the image into rgb values
    if image == 'gym':
        orginal_image = plt.imread(r'test images\Orginal image - gym.jpeg')
        image_path = 'test images\Orginal image - gym.jpeg'

    if image == 'goosefair':
        orginal_image = plt.imread(r'test images\Orginal image - goosefair.jpeg')
        image_path = 'test images\Orginal image - goosefair.jpeg'
    
    if image == 'goosefair bright':
        orginal_image = plt.imread(r'test images\Orginal image - goosefair bright.jpeg')
        image_path = 'test images\Orginal image - goosefair bright.jpeg'

    if image == 'edit goosefair bright':
        orginal_image = plt.imread(r'test images\edit image - goosefair bright.jpeg')
        image_path = 'test images\edit image - goosefair bright.jpeg'

    if image == 'goosefair day':
        orginal_image = plt.imread(r'test images\Orginal image - goosefair day.jpeg')

        image_path = 'test images\Orginal image - goosefair.jpeg'

    if image == 'ghost':
        orginal_image = plt.imread(r'test images\Orginal image - ghost.jpeg')
        image_path = 'test images\Orginal image - ghost.jpeg'

    if image == 'ocean':
        orginal_image = plt.imread(r'test images\Orginal image - ocean.jpeg')
        image_path = 'test images\Orginal image - ocean.jpeg'

    if image == 'lake':
        orginal_image = plt.imread(r'test images\Orginal image - lakejpeg.jpeg')
        image_path = 'test images\Orginal image - lakejpeg.jpeg'

    if image == 'orginal':
        orginal_image = plt.imread(r'test images\Orginal image.jpeg')
        image_path = 'test images\Orginal image.jpeg'

    if image == 'webcam':
        orginal_image = plt.imread(r'test images\WIN_20221101_14_24_12_Pro.jpg')
        image_path = 'test images\WIN_20221101_14_24_12_Pro.jpg'

    if image == 'fake moose':
        orginal_image = plt.imread(r'test images\fake moose.jpg')
        image_path = r'test images\fake moose.jpg'

    if image == 'fake lion':
        orginal_image = plt.imread(r'test images\fake lion.jpg')
        image_path = r'test images\fake lion.jpg'

    if image == 'fake lion small':
        orginal_image = plt.imread(r'test images\fake lion small.jpg')
        image_path = r'test images\fake lion small.jpg'
    
    if image == 'fake lion resampled':
        orginal_image = plt.imread(r'test images\fake lion resampled.jpg')
        image_path = r'test images\fake lion resampled.jpg'

    if image == 'real zebra':
        orginal_image = plt.imread(r'test images\real zebra.jpg')
        image_path = r'test images\real zebra.jpg'

    if image == 'real zebra resampled':
        orginal_image = plt.imread(r'test images\real zebra resampled.jpg')
        image_path = r'test images\real zebra resampled.jpg'


    if image == 'real bird':
        orginal_image = plt.imread(r'test images\real bird.jpg')
        image_path = r'test images\real bird.jpg'
    return orginal_image, image_path
