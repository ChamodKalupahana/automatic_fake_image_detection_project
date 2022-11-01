import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance
import time

start_time = time.time()

def SPN():
    
    return

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image


fig, ax = plt.subplots(1, 2)

image_path = 'test images\edit image - goosefair bright.jpeg'
#image_path = 'test images\WIN_20221101_14_24_12_Pro.jpg'
ela_image = convert_to_ela_image(image_path, 90)

org_image = plt.imread(image_path)

ax[0].imshow(org_image)
ax[1].imshow(ela_image)
plt.savefig('test image.jpeg')
plt.show()

# measure time taken to execute code (uni interpreter is usually faster than uni_2_1)
end_time = time.time()
time_taken = end_time - start_time
print('Time taken =', str(time_taken) + 'secs')