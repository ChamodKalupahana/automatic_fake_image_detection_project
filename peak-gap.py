import numpy as np
import matplotlib.pyplot as plt
import time
from loading_data import load_image

start_time = time.time()

def brighten_greyscale_image(orginal_image):
    r, g, b = orginal_image[:,:,0], orginal_image[:,:,1], orginal_image[:,:,2]
    greyscale_image = 0.2989 * r + 0.5870 * g + 0.1140 * b

    greyscale_image = (greyscale_image / np.amax(greyscale_image)).astype(np.float64)
    
    alpha=10
    #edit_greyscale_image = np.log(1 + alpha * greyscale_image) / np.log(1 + alpha)
    edit_greyscale_image = np.log(1 + alpha * greyscale_image) / np.log(1 + alpha)

    return greyscale_image, edit_greyscale_image

def find_grey_hist(orginal_image):

    greyscale_image, edit_greyscale_image = brighten_greyscale_image(orginal_image)
    
    hist, bin_edges = np.histogram(greyscale_image * 256, bins=100, range=(0, 256))
    hist_edit, bin_edges_edit = np.histogram(edit_greyscale_image * 256, bins=100, range=(0, 256))

    # zero bin counts
    # this function returns x values in the 1st array and y values in the 2nd array
    orginal_zero_bins = np.where(greyscale_image == 0)

    orginal_bins = np.where(hist == 0)
    edit_bins = np.where(hist_edit == 0) # might be no black pixels



    #neighboring_bins = edit_greyscale_image[edit_zero_bins[0] + 1, edit_zero_bins[1] + 1]
        
    
    # need to test images to figure out what threshold should be
    # need program to return a confidence value
    # confidence value should depend on the difference between neighbouring bins
    
    neighboring_bin_threshold = 0
    if (hist_edit[int(edit_bins + 1)] > neighboring_bin_threshold) and :

    confidence_level = 

    if neighboring_bins > 0.5:
        fake = True

    if neighboring_bins <= 0.5:
        fake = False


    return hist, bin_edges, hist_edit, bin_edges_edit


#------------------------------------------------------------------------
#--------------------- Fake Image Calculations ---------------------
#------------------------------------------------------------------------

# load_image() is a function that I made and defined earlier
# fake moose, fake lion, real zebra, real bird are downloaded images from CASIA database
orginal_image, image_path = load_image('real zebra')

width, height = np.shape(orginal_image)[0], np.shape(orginal_image)[1]

greyscale_image, edit_greyscale_image = brighten_greyscale_image(orginal_image)
hist, bin_edges, hist_edit, bin_edges_edit = find_grey_hist(orginal_image)

num_of_pixels = width * height
print('Resolution = '+ str(width) + 'x' + str(height))

# calculate probabilty of being fake
fakeness = np.sum(orginal_image) / (num_of_pixels * 255)
print('probabilty of fake = {fakeness:.2f} %'.format(fakeness=fakeness * 100)) # in percentage

#------------------------------------------------------------------------
#--------------------- Plotting Infomation ---------------------
#------------------------------------------------------------------------

fig, ax = plt.subplots(1, 2)

ax[0].imshow(greyscale_image, cmap='gray')
ax[1].imshow(edit_greyscale_image, cmap='gray')

ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xticks([])
ax[1].set_yticks([])

ax[0].set_title('Greyscale of orginal Image')
ax[1].set_title('Brightened Greyscale')

fig_2, ax_2 = plt.subplots(1, 2)

ax_2[0].bar(bin_edges[0:-1], hist, align='edge', width=20)
ax_2[1].bar(bin_edges_edit[0:-1], hist_edit, align='edge', width=20)

ax_2[0].set_title('Histrogram of orginal image')
ax_2[1].set_title('histrogram of brightened image')

plt.savefig("figures/test image peak-gap hist")
plt.show()

# measure time taken to execute code (uni interpreter is usually faster than uni_2_1)
end_time = time.time()
time_taken = end_time - start_time
print('Time taken =', str(time_taken) + 'secs')