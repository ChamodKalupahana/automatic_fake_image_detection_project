import numpy as np
import matplotlib.pyplot as plt
import time

from resampling import detect_resampling
import glob
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


start_time = time.time()


def fake_image_detection(show_point_plot, show_scatter, fake_threshold, show_truth_table):
    # how many images to process
    casia_folder_path = r"C:\Users\chamo\Documents\Physics\Projects\Imaging and Data Processing\Automatic Fake Image Detection"

    num_of_images = 300
    casia_2_real = glob.glob(casia_folder_path+'\CASIA2\Au\Au*')
    casia_2_fake = glob.glob(casia_folder_path+'\CASIA2\Tp\Tp*')
    casia_2 = np.append(casia_2_real, casia_2_fake)

    total_results = np.array([])
    fake_image_labels = np.array([])
    fakeness_end_result_total = np.array([])

    for i in range(num_of_images):
        # get a radnom real or fake image
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

        if '\CASIA2\Tp\Tp' in str(temp_image):
            fake_image = True
            fake_image_labels = np.append(fake_image_labels, fake_image)
            print('Image', str(i + 1),'is Fake')

        if fakeness_end_result == fake_image:
            total_results = np.append(total_results, 1)
        
        if fakeness_end_result != fake_image:
            total_results = np.append(total_results, 0)
        
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

    if show_scatter == True:
        bar_chart_results = np.empty(2)
        bar_chart_results[0] = np.sum(total_results)
        bar_chart_results[1] = num_of_images - bar_chart_results[0]

        bar_chart_array = np.array(['Correct', 'Wrong'])

        plt.figure()
        plt.title('Chart of how many images were detected correctly')
        plt.bar(bar_chart_array, bar_chart_results)
        plt.savefig(r'figures\bar chart.jpeg')
    
    if show_truth_table == True:
        #labels = np.array(['RF', 'FF', 'RR', 'FR'])
        labels = [['Real', 'Fake']]
        labels = [0, 1]
        
        #truth = confusion_matrix(total_results, fake_image_labels)
        #truth_table = ConfusionMatrixDisplay(confusion_matrix=truth)

        print(np.size(np.where(fakeness_end_result_total == fake_image_labels)))

        # detect_resampling returns 1 if fake
        # fake_image_labels returns 1 if fake
        truth = confusion_matrix(fake_image_labels, fakeness_end_result_total, labels=labels)
        truth_table = ConfusionMatrixDisplay(confusion_matrix=truth, display_labels=labels)
        
        #truth_table.set_ylabel('Program output')
        #truth_table.set_xlabel('Image')
        truth_table.plot()
        plt.savefig(r'figures\truth table.jpeg')


    # calculate accuracy of system
    accuracy = np.sum(total_results) / np.size(total_results)
    print('Fake Image Detection Accuracy = {accuracy:.2f}%'.format(accuracy=accuracy * 100)) # in percent

    plt.show()
    
    return

# with threshold 0.1, the results are very 'pure'
# most/half of fake images that the program detects is correct but it misses 75% of fake images in sample
fake_image_detection(show_point_plot=False, show_scatter=True, fake_threshold=0.2, show_truth_table=True)
#detect_resampling(suspect_image=casia_2_fake[10], show_images=True, debugging=False)

# measure time taken to execute code (uni interpreter is usually faster than uni_2_1)
end_time = time.time()
time_taken = end_time - start_time
print('Time taken =', str(time_taken) + 'secs')

# resolution of all images = 256x384