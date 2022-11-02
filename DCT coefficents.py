import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.fftpack import dct
from loading_data import load_image


start_time = time.time()

def dct_coefficents(orginal_image):
    dct_image = dct(dct(orginal_image, axis=0), axis=1)
    
    return dct_image

#------------------------------------------------------------------------
#--------------------- Fake Image Calculations ---------------------
#------------------------------------------------------------------------

# fake moose, fake lion, real zebra, real bird
orginal_image, image_path = load_image('fake moose')

# make greyscale image
r, g, b = orginal_image[:,:,0], orginal_image[:,:,1], orginal_image[:,:,2]
greyscale_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
greyscale_image = (greyscale_image / np.amax(greyscale_image)).astype(np.float64)

dct_image = dct(greyscale_image)

width, height = np.shape(orginal_image)[0], np.shape(orginal_image)[1]

num_of_pixels = width * height
print('Resolution = '+ str(width) + 'x' + str(height))

# calculate histrogram
hist, bin_edges = np.histogram(greyscale_image * 256, bins=100, range=(0, 256))

#------------------------------------------------------------------------
#--------------------- Plotting Infomation ---------------------
#------------------------------------------------------------------------

fig, ax = plt.subplots(1, 2)

ax[0].imshow(orginal_image)
ax[1].imshow(dct_image, cmap='gray')

ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xticks([])
ax[1].set_yticks([])

ax[0].set_title('Orginal Image')
ax[1].set_title('DCT Image')

# plot histrogram
fig_2, ax_2 = plt.subplots(1)

ax_2.bar(bin_edges[0:-1], hist, align='edge', width=20)

ax_2.set_title('Histrogram of DCT Image')
ax_2.set_ylabel('Num. of pixels')
ax_2.set_xlabel('Pixel Brightess')

fig.savefig(r'figures\test image DCT.jpeg')
fig_2.savefig(r'figures\test image hist DCT.jpeg')
plt.show()

# measure time taken to execute code (uni interpreter is usually faster than uni_2_1)
end_time = time.time()
time_taken = end_time - start_time
print('Time taken =', str(time_taken) + 'secs')