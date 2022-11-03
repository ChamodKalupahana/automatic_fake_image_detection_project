import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.fftpack import dct
from loading_data import load_image


start_time = time.time()

def get_blocks(orginal_image, width, height):
    # define 3x3 blocks
    block_width = 2
    block_height = 2

    block_array = np.empty((width, height, block_width, block_width))
    
    # loop over the whole image
    # this step will computationally intense so we loaded in a smaller 106 x 156 image
    for x in range(width - 1):
        for y in range(height - 1):
            new_block_array = orginal_image[x : x + block_width, y : y + block_height]
            print(str(x), str(y))
            #block_array = np.append(block_array, new_block_array)
            block_array[x, y] = new_block_array
    
    #block_array = np.reshape(block_array, (width, height, block_width, block_height))
    return block_array, block_width, block_height

def dct_coefficents(orginal_image, width, height):
    block_array = get_blocks(orginal_image, width, height)
    
    dct_image = np.np.empty((width, height, block_width, block_width))
    # apply dct to each block
    for x in range(width):
        for y in range(height):
                new_dct_image = dct(dct(block_array[x][y], axis=0), axis=1)
    
    return dct_image

#------------------------------------------------------------------------
#--------------------- Fake Image Calculations ---------------------
#------------------------------------------------------------------------

# fake moose, fake lion, fake lion small, real zebra, real bird
orginal_image, image_path = load_image('fake lion small')

# make greyscale image
r, g, b = orginal_image[:,:,0], orginal_image[:,:,1], orginal_image[:,:,2]
greyscale_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
greyscale_image = (greyscale_image / np.amax(greyscale_image)).astype(np.float64)


width, height = np.shape(orginal_image)[0], np.shape(orginal_image)[1]

dct_image = dct_coefficents(greyscale_image, width, height)

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