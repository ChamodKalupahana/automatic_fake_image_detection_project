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
    
    fakeness = np.mean(win_var) / 250
    return fakeness

def get_variance_v2(orginal_image, width, height, test_variance):

    if test_variance == True:
        orginal_image = np.random.rand(width, height) * 255


    image_mean = np.mean(orginal_image)
    total_variance = np.array([])
    """

    for x in range(width):
        for y in range(height):
            variance = (orginal_image[x][y] - image_mean)**2
            total_variance = np.append(total_variance, variance)
            print("Processing pixel", str(x), str(y))
    """
    var_array = np.copy(orginal_image)
    var_array = (orginal_image - image_mean)**2
    total_var = np.sqrt(np.sum(var_array) / (width * height) - 300)
    
    #image_variance = np.sqrt(np.sum(total_variance / (width * height)) - 300)
    image_variance = total_var
    
    # max variance is 5000?
    # max variance should be 255 because that's the max brightness of the pixels in the fft_image

    # found by 'training' the program using sample real and fake images
    fake_threshold = 0.15
    if image_variance > image_mean:
        fakeness = 1
    
    # extravagate small differences
    if image_variance <= image_mean:
        fakeness = (image_variance / image_mean) * 3

    if image_variance < (image_mean * fake_threshold):
        #fakeness = 0
        fakeness = image_variance / image_mean
    
    return fakeness

def get_max(orginal_image, width, height, test_variance):

    fakeness = np.max(orginal_image) / np.sum(orginal_image)
    
    return fakeness * 10000

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

# fake moose, fake lion, fake lion small, fake lion resampled, real zebra, real zebra resampled, real zebra small, real bird
orginal_image, image_path = load_image('real zebra small')

# make greyscale image
r, g, b = orginal_image[:,:,0], orginal_image[:,:,1], orginal_image[:,:,2]
greyscale_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
greyscale_image = (greyscale_image / np.amax(greyscale_image)).astype(np.float64)

width, height = np.shape(orginal_image)[0], np.shape(orginal_image)[1]

fft_image = fft_coefficents(greyscale_image, enchance=True, high_pass_filter=False)

num_of_pixels = width * height
print('Resolution = '+ str(width) + 'x' + str(height))

fakeness = get_variance_v2(fft_image, width, height, test_variance=False)
#fakeness = get_max(fft_image, width, height, test_variance=False)

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