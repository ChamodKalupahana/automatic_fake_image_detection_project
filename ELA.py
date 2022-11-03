from json import load
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance
import time
from loading_data import load_image

start_time = time.time()

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

#------------------------------------------------------------------------
#--------------------- Fake Image Calculations ---------------------
#------------------------------------------------------------------------

# load_image() is a function that I made and defined earlier
# fake moose, fake lion, real zebra, real bird are downloaded images from CASIA database
orginal_image, image_path = load_image('fake lion')

# higher ela brightness means more real?
# same level of ela brightness throughout image means more real

ela_image = np.asarray(convert_to_ela_image(image_path, 90))

width, height = np.shape(orginal_image)[0], np.shape(orginal_image)[1]

num_of_pixels = width * height
print('Resolution = '+ str(width) + 'x' + str(height))

# calculate probabilty of being fake
fakeness = np.sum(ela_image) / (num_of_pixels * 255)
print('probabilty of fake = {fakeness:.2f} %'.format(fakeness=fakeness * 100)) # in percentage

# calculate histrogram
r, g, b = ela_image[:,:,0], ela_image[:,:,1], ela_image[:,:,2]
greyscale_ela_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
greyscale_ela_image = (greyscale_ela_image / np.amax(greyscale_ela_image)).astype(np.float64)

hist_ela, bin_edges_ela = np.histogram(greyscale_ela_image * 256, bins=100, range=(0, 256))

#------------------------------------------------------------------------
#--------------------- Plotting Infomation ---------------------
#------------------------------------------------------------------------

fig, ax = plt.subplots(1, 2)

ax[0].imshow(orginal_image)
ax[1].imshow(ela_image)

ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xticks([])
ax[1].set_yticks([])

ax[0].set_title('Orginal Image')
ax[1].set_title('ELA Image')

# plot histrogram
fig_2, ax_2 = plt.subplots(1)

ax_2.bar(bin_edges_ela[0:-1], hist_ela, align='edge', width=20)

ax_2.set_title('Histrogram of ELA Image')
ax_2.set_ylabel('Num. of pixels')
ax_2.set_xlabel('Pixel Brightess')

fig.savefig(r'figures\test image ELA.jpeg')
fig_2.savefig(r'figures\test image hist ELA.jpeg')
plt.show()

# measure time taken to execute code (uni interpreter is usually faster than uni_2_1)
end_time = time.time()
time_taken = end_time - start_time
print('Time taken =', str(time_taken) + 'secs')