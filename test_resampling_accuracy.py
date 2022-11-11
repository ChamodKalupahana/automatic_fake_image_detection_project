import numpy as np
import matplotlib.pyplot as plt
import time

from detect_resampling import fake_image_detection

def test_accuracy_against_k(min_k, max_k, casia_folder_path, defacto_folder_path):
    #fake_threshold = np.arange(0.15, 0.3, 0.005)
    fake_threshold = np.linspace(min_k, max_k, 200)
    #fake_threshold = 0.2

    accuracy = np.array([])

    # test fake_image_detection for varying k values
    for i in range(0, np.size(fake_threshold)):
        temp_accuracy = fake_image_detection(fake_threshold=fake_threshold[i], num_of_images=300, show_bar_chart=False, show_point_plot=False, show_truth_table=False, produce_half_sample=True, casia_folder_path=casia_folder_path, defacto_folder_path=defacto_folder_path)
        # produce_half_sample = False here because we want to test varying fake_threshold against the same images
        # accuracy plot can have sharp edges if num_of_images isn't high enough

        # for 200 k values and 300 images per iteration, program runs through 60,000 images
        accuracy = np.append(accuracy, temp_accuracy)
        print('Fake Threshold (k) = {fake_threshold:.3f} done'.format(fake_threshold=fake_threshold[i]))

    # plotting infomation
    plt.figure()
    plt.plot(fake_threshold, accuracy * 100, 'r-')
    plt.xlabel('Fake Threshold (k)')
    plt.ylabel('Accuracy (%)')
    plt.savefig(r'figures\accuracy vs fake threshold plot.jpg', dpi=300)

    # max accuracy occurs at 0.675585
    # at index 84, 85 and 105
    # k1 = 0.20110552763819095
    # k2 = 0.201356783919598
    # k3 = 0.2063819095477387

    #print('Highest accuracy of {accuracy:.4f} occurs at {fake_threshold:.4f}'.format(accuracy=accuracy, fake_threshold=fake_threshold))
    plt.show()

    return np.max(accuracy)

#test_accuracy_against_k(min_k=0.18, max_k=0.23, casia_folder_path=casia_folder_path, defacto_folder_path=defacto_folder_path)