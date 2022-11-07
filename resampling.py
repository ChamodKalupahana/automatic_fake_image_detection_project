import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.fftpack import dct
from loading_data import load_image

from scipy import ndimage 

start_time = time.time()

def get_variance(orginal_image, width, height):
    win_rows, win_cols = 5, 5

    img = orginal_image
    win_mean = ndimage.uniform_filter(img, (win_rows, win_cols))
    win_sqr_mean = ndimage.uniform_filter(img**2, (win_rows, win_cols))
    win_var = win_sqr_mean - win_mean**2
    
    return np.var(win_var)

def fft_coefficents(orginal_image, enchance, high_pass_filter):
    #block_array = get_blocks(orginal_image, width, height)
    
    fft_image = np.fft.fft2(orginal_image)
    if enchance == True:
        freq_image = 20*np.log(np.fft.fftshift(fft_image))
    if enchance == False:
        freq_image = np.fft.fftshift(fft_image)
    
    if high_pass_filter == True:
        (w, h) = orginal_image.shape
        half_w, half_h = int(w/2), int(h/2)

        # high pass filter
        n = 50
        freq_image[half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0

    return np.abs(freq_image)

#------------------------------------------------------------------------
#--------------------- Fake Image Calculations ---------------------
#------------------------------------------------------------------------

# fake moose, fake lion, fake lion small, fake lion resampled, real zebra, real bird
orginal_image, image_path = load_image('goosefair')

# make greyscale image
r, g, b = orginal_image[:,:,0], orginal_image[:,:,1], orginal_image[:,:,2]
greyscale_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
greyscale_image = (greyscale_image / np.amax(greyscale_image)).astype(np.float64)


width, height = np.shape(orginal_image)[0], np.shape(orginal_image)[1]

fft_image = fft_coefficents(greyscale_image, enchance=True, high_pass_filter=False)

num_of_pixels = width * height
print('Resolution = '+ str(width) + 'x' + str(height))

fakeness = get_variance(fft_image, width, height)

print('probabilty of fake = {fakeness:.2f} %'.format(fakeness=fakeness * 100)) # in percentage

# calculate histrogram
hist, bin_edges = np.histogram(greyscale_image * 256, bins=100, range=(0, 256))

#------------------------------------------------------------------------
#--------------------- Plotting Infomation ---------------------
#------------------------------------------------------------------------

fig, ax = plt.subplots(1, 2)

ax[0].imshow(orginal_image)
ax[1].imshow(fft_image, cmap='gray')

ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xticks([])
ax[1].set_yticks([])

ax[0].set_title('Orginal Image')
ax[1].set_title('Fourier Transform Image')

fig.savefig(r'figures\test image resampling.jpeg')
"""
# plot histrogram
fig_2, ax_2 = plt.subplots(1)

ax_2.bar(bin_edges[0:-1], hist, align='edge', width=20)

ax_2.set_title('Histrogram of DCT Image')
ax_2.set_ylabel('Num. of pixels')
ax_2.set_xlabel('Pixel Brightess')


fig_2.savefig(r'figures\test image hist DCT.jpeg')

"""
plt.show()
# measure time taken to execute code (uni interpreter is usually faster than uni_2_1)
end_time = time.time()
time_taken = end_time - start_time
print('Time taken =', str(time_taken) + 'secs')