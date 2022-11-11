import numpy as np
import matplotlib.pyplot as plt
import time

from resampling import detect_resampling
import glob

# for truth table
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

start_time = time.time()

def fake_image_detection(show_point_plot, show_bar_chart, fake_threshold, show_truth_table, produce_half_sample, num_of_images):
    # how many images to process
    casia_folder_path = r"C:\Users\chamo\Documents\Physics\Projects\Imaging and Data Processing\Automatic Fake Image Detection"
    defacto_folder_path = r"C:\Users\chamo\Documents\Physics\Projects\Imaging and Data Processing\Automatic Fake Image Detection\copymove_img\img"

    # max num. of images for real + crop is 14982
    # max num. of images for real + fake is 11429
    # max num. of images for real + copy-move is 13749
    # use 300 images for testing

    #num_of_images = 11429
    casia_2_real = glob.glob(casia_folder_path+'\CASIA2\Au\Au*')
    casia_2_fake = glob.glob(casia_folder_path+'\CASIA2\Tp\Tp*')
    casia_2_real_crop = glob.glob(casia_folder_path+'\CASIA2_RandomCrop\Au\Au*')
    defacto = glob.glob(defacto_folder_path+r'\0_*.tif')
    
    if produce_half_sample == True:
        casia_2 = np.append(casia_2_real[0:int(np.round(num_of_images/2))], casia_2_real_crop[0:int(np.round(num_of_images/2))])
        #casia_2 = np.append(casia_2_real[0:int(np.round(num_of_images/2))], casia_2_fake[0:int(np.round(num_of_images/2))])
        #casia_2 = np.append(casia_2_real[0:int(np.round(num_of_images/2))], defacto[0:int(np.round(num_of_images/2))])


    if produce_half_sample == False:
        casia_2 = np.append(casia_2_real, casia_2_real_crop)
        #casia_2 = np.append(casia_2_real, casia_2_fake)
        #casia_2 = np.append(casia_2_real, defacto)

    total_results = np.array([])
    fake_image_labels = np.array([])
    fakeness_end_result_total = np.array([])

    good_images = np.array([])
    bad_images = np.array([])

    for i in range(num_of_images - 1):
        # go through all images 1 by 1
        if produce_half_sample == True:
            random_i = i
        
        # get a random real or fake image
        if produce_half_sample == False:
            random_i = np.random.randint(0, np.size(casia_2))

        temp_image = casia_2[random_i]

        # fake_threshold of 0.15 was too low to detect real images (accuracy = 40% for just real images)
        # fake_threshold of 0.2 was too high to detect fake images (accuracy = 30% for just fake images)
        # make a plot of different thresholds and accurary of program

        fakeness_end_result = detect_resampling(suspect_image=temp_image, show_images=False, debugging=False, fake_threshold=fake_threshold)

        if '\CASIA2\Au\Au' in str(temp_image):
            fake_image = False
            fake_image_labels = np.append(fake_image_labels, fake_image)
            print('Image', str(i + 1),'is Real')

        if '\CASIA2_RandomCrop\Au\Au' in str(temp_image):
            fake_image = True
            fake_image_labels = np.append(fake_image_labels, fake_image)
            print('Image', str(i + 1),'is Cropped')

        if '\CASIA2\Tp\Tp' in str(temp_image):
            fake_image = True
            fake_image_labels = np.append(fake_image_labels, fake_image)
            print('Image', str(i + 1),'is Fake')

        if '\copymove_img\img' in str(temp_image):
            fake_image = True
            fake_image_labels = np.append(fake_image_labels, fake_image)
            print('Image', str(i + 1),'is Copy-move')

        if fakeness_end_result == fake_image:
            total_results = np.append(total_results, 1)
            good_images = np.append(good_images, str(temp_image))
        
        if fakeness_end_result != fake_image:
            total_results = np.append(total_results, 0)
            bad_images = np.append(bad_images, str(temp_image))
        
        fakeness_end_result_total = np.append(fakeness_end_result_total, fakeness_end_result)

    # for plotting
    if show_point_plot == True:
        image_index = np.arange(0, num_of_images)

        plt.figure()
        plt.title('Scatter plot of what images where detected correctly')
        plt.ylabel('1 for correct, 0 for wrong')
        plt.xlabel('Image index')
        plt.plot(image_index, total_results, 'b*')
        plt.savefig(r'figures\scatter plot.jpeg')

    if show_bar_chart == True:
        plt.figure()
        
        bar_chart_results = np.empty(2)
        bar_chart_results[0] = np.sum(total_results)
        bar_chart_results[1] = num_of_images - bar_chart_results[0]
        plt.ylabel('Num. of images')

        # for percentage 
        bar_chart_results[0] = (bar_chart_results[0] / num_of_images) * 100
        bar_chart_results[1] = (bar_chart_results[1] / num_of_images) * 100
        plt.ylabel('Percentage (%)')

        bar_chart_array = np.array(['Correct', 'Wrong'])

        #plt.title('Chart of how many images were detected correctly')
        plt.bar(bar_chart_array, bar_chart_results)
        plt.savefig(r'figures\bar chart.jpeg')
    
    if show_truth_table == True:
        print('Correct Answers: ' + str(np.size(np.where(fakeness_end_result_total == fake_image_labels))))

        #labels = np.array(['RF', 'FF', 'RR', 'FR'])
        str_labels = ['Real', 'Fake'] # figure out how to fix labels
        labels = [0, 1]

        # detect_resampling returns 1 if fake
        # fake_image_labels returns 1 if fake
        
        # to put real and fake labels
        #fig_truth, ax_truth = plt.subplots(1)
        #truth = confusion_matrix(fake_image_labels, fakeness_end_result_total , labels=labels, normalize='true')
        #truth_table = ax_truth.matshow(truth, cmap='winter')
        #fig_truth.colorbar(truth_table)
        #ax_truth.set_xticklabels([''] + str_labels)
        #ax_truth.set_yticklabels([''] + str_labels)

        # to put nummbers on table
        truth = confusion_matrix(fake_image_labels, fakeness_end_result_total, normalize='all')
        truth_table = ConfusionMatrixDisplay(confusion_matrix=truth, display_labels=labels)
        cmap=["lightgreen", "mistyrose", "lightyellow", "lightgray"]
        truth_table.plot(cmap='RdYlGn') # use RdYlGn or binary or OrRd_r
        
        #truth_table.set_ylabel('Program output')

        #truth_table.set_xlabel('Image')
        #plt.show()
        plt.savefig(r'figures\truth table.jpeg')


    # calculate accuracy of system
    accuracy = np.sum(total_results) / np.size(total_results)
    percentage = np.sum(fake_image_labels) / num_of_images
    print('Percentage of fake images = {percentage:.2f}%'.format(percentage=percentage * 100))
    print('Fake Image Detection Accuracy = {accuracy:.2f}%'.format(accuracy=accuracy * 100)) # in percent

    plt.show()
    
    #plt.imshow(plt.imread(good_images[0])) for debugging
    # Au_ani_00005 is guessed correctly by program for both real and resized

    # orginal theory was that the program can't detect if the image wasn't resized by a significant factor
    # but bad_images contains images which have been cropped in significantly
    return accuracy

# with threshold 0.1, the results are very 'pure'
# most/half of fake images that the program detects is correct but it misses 75% of fake images in sample

# parameter testing
# confidence = 0.5, fake_threshold = 0.1, variance_mutiplier = 3 works - accurary = 55% on average (pure sample)
# confidence = 0.5, fake_threshold = 0.1, variance_mutiplier = 3.5 works - accurary = 55.6% on average
# confidence = 0.4, fake_threshold = 0.01, variance_mutiplier = 2 doesn't work - accuracy = 48.7%

# confidence = 0.5, fake_threshold = 0.1, variance_mutiplier = 3.5, const_variance = -300 works - accurary = 48% on average
# confidence = 0.5, fake_threshold = 0.2, variance_mutiplier = 3.5, const_variance = -500 works - accurary = 59.6% on average
# confidence = 0.5, fake_threshold = 0.2, variance_mutiplier = 4, const_variance = -450 works - accurary = 59.6% on average
# confidence = 0.5, fake_threshold = 0.175, variance_mutiplier = 3.5, const_variance = -450 works - accurary = 59.6% on average
# confidence = 0.5, fake_threshold = 0.2, variance_mutiplier = 3.5, const_variance = -475 works - accurary = 64% on average
# confidence = 0.5, fake_threshold = 0.2, variance_mutiplier = 3.5, const_variance = -485 works - accurary = 65% on average
# confidence = 0.5, fake_threshold = 0.2, variance_mutiplier = 3.5, const_variance = -490 works - accurary = 65.7% on average
# confidence = 0.5, fake_threshold = 0.2, variance_mutiplier = 3.5, const_variance = -495 works - accurary = 66.3% on average
# confidence = 0.5, fake_threshold = 0.2, variance_mutiplier = 3.5, const_variance = -500 works - accurary = 66.3% on average

# const_variance = -495 is the global maximum
# variance_mutiplier = 3.5 is the global maximum
# fake_threshold = 0.2 is the global maximum
# confidence = 0.5 is the global maximum

#fake_image_detection(show_point_plot=False, show_bar_chart=True, fake_threshold=0.2, show_truth_table=True, produce_half_sample=False,
#num_of_images=13749)

# for a single image anaylsis
#detect_resampling(suspect_image=casia_2_fake[10], show_images=True, debugging=False)
#detect_resampling(suspect_image=defacto[10], show_images=True, debugging=False)

# measure time taken to execute code (uni interpreter is usually faster than uni_2_1)
end_time = time.time()
time_taken = end_time - start_time
print('Time taken = {time_taken:.2f} secs'.format(time_taken=time_taken))

# resolution of all images = 256x384 (most images are this size)