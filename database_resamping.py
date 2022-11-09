import numpy as np
import matplotlib.pyplot as plt

from loading_data import load_image
import glob

import albumentations as A


def test_random_crop(suspect_image):
    """ Randomly crops a sample image and shows orginal against cropped

    Args:
        suspect_image (str): String input into loading_data.py
    """
    orginal_image, image_path = load_image(suspect_image)
    width, height = np.shape(orginal_image)[0], np.shape(orginal_image)[1]

    # resolution of all images in casia_v2 = 256x384
    random_width = np.random.randint(100, width)
    random_height = np.random.randint(100, height)

    # Declare an augmentation pipeline
    # Crop into a random section of the image with random dimensions
    transform = A.Compose([
        # width and height are swapped around here???? in the albumentations library
        A.RandomCrop(width=random_height, height=random_width),
    ])

    transformed = transform(image=orginal_image)
    transformed_image = transformed['image']

    # plotting infomation
    plt.figure()
    plt.imshow(transformed_image)
    plt.title('Randomly Cropped Image')

    plt.figure()
    plt.imshow(orginal_image)
    plt.title('Orginal Image')
    plt.show()

    return

def random_crop_database():
    """ Randomly crops Casia_v2 real images and saves them into a separate folder

    Args:
        suspect_image (str): String input into loading_data.py
    """
    
    casia_folder_path = r"C:\Users\chamo\Documents\Physics\Projects\Imaging and Data Processing\Automatic Fake Image Detection"
    casia_edited_folder_path = r"C:\Users\chamo\Documents\Physics\Projects\Imaging and Data Processing\Automatic Fake Image Detection"
    casia_2_real = glob.glob(casia_folder_path+'\CASIA2\Au\Au*')

    casia_2 = casia_2_real
    num_of_images = np.size(casia_2_real)

    for i in range(0, num_of_images):
        temp_image_image_path = casia_2[i]
        temp_image = plt.imread(temp_image_image_path)

        width, height = np.shape(temp_image)[0], np.shape(temp_image)[1]

        # resolution of all images in casia_v2 = 256x384 or 384x256
        random_width = np.random.randint(100, width)
        random_height = np.random.randint(100, height)

        # Declare an augmentation pipeline
        # Crop into a random section of the image with random dimensions
        transform = A.Compose([
            # width and height are swapped around here???? in the albumentations library
            A.RandomCrop(width=random_height, height=random_width),
        ])

        transformed = transform(image=temp_image)
        transformed_image = transformed['image']
        
        # gets the 'file name' of the image e.g ani_00001.jpg
        temp_image_file_name = temp_image_image_path[-14:-4]

        # saves the images with the same file name with (recrop) in a different folder at the end
        plt.imsave(casia_edited_folder_path + r"\CASIA2_RandomCrop\Au\Au_" + temp_image_file_name + r'_recrop.jpg', transformed_image)

        # what happens if there's already files with the same name there? (doesn't matter, over-writes)
        print('Loop '+ str(i) +': ' + str(temp_image_file_name) + ' saved')

    return


#test_random_crop(suspect_image='fake moose')
random_crop_database()