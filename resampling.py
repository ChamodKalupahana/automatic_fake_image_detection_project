import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.fftpack import dct
from loading_data import load_image

from scipy import ndimage 

start_time = time.time()
def detect_resampling(suspect_image ,show_images, debugging, fake_threshold):

    def get_variance_v2(orginal_image, width, height, test_variance, fake_threshold):

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
        
        # need magnitude of variance because it can be -ve for some edge case images (don't know why)
        # happened with CASIA2\Tp\Tp_D_CNN_M_B_nat10139_nat00059_11949.jpg
        total_var = np.sqrt(np.abs(np.sum(var_array) / (width * height) - 300)) 
        
        #image_variance = np.sqrt(np.sum(total_variance / (width * height)) - 300)
        image_variance = total_var
        
        # max variance is 5000?
        # max variance should be 255 because that's the max brightness of the pixels in the fft_image

        # found by 'training' the program using sample real and fake images
        # use fake_threshold from outisde this function

        if image_variance > image_mean:
            fakeness = 1
        
        # extravagate small differences
        if image_variance <= image_mean:
            #fakeness = (image_variance / image_mean) * 3
            fakeness = (image_variance / image_mean) * 3.5

        if image_variance < (image_mean * fake_threshold):
            #fakeness = 0
            fakeness = image_variance / image_mean
        
        return fakeness

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
    if debugging == False:
        orginal_image= plt.imread(suspect_image)

    # fake moose, fake lion, fake lion small, fake lion resampled, real zebra, real zebra resampled, real zebra small, real bird    
    if debugging == True:
        orginal_image, image_path = load_image(suspect_image)

    # make greyscale image
    r, g, b = orginal_image[:,:,0], orginal_image[:,:,1], orginal_image[:,:,2]
    greyscale_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
    greyscale_image = (greyscale_image / np.amax(greyscale_image)).astype(np.float64)

    width, height = np.shape(orginal_image)[0], np.shape(orginal_image)[1]

    fft_image = fft_coefficents(greyscale_image, enchance=True, high_pass_filter=False)

    num_of_pixels = width * height
    #print('Resolution = '+ str(width) + 'x' + str(height))

    fakeness = get_variance_v2(fft_image, width, height, test_variance=False, fake_threshold=fake_threshold)
    #fakeness = get_max(fft_image, width, height, test_variance=False)

    print('probabilty of fake = {fakeness:.2f} %'.format(fakeness=fakeness * 100)) # in percentage
    
    if show_images == True:
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
        plt.show()

    #confidence = 0.25
    confidence = 0.5
    if fakeness > confidence:
        fakeness_end_result = True # image is fake

    if fakeness <= confidence:
        fakeness_end_result = False # image is real
    
    return fakeness_end_result

#detect_resampling(suspect_image='fake moose', show_images=True, debugging=True, fake_threshold=0.2)

# measure time taken to execute code (uni interpreter is usually faster than uni_2_1)
end_time = time.time()
time_taken = end_time - start_time
#print('Time taken =', str(time_taken) + 'secs')