import numpy as np
import matplotlib.pyplot as plt
import time

from detect_resampling import fake_image_detection
from resampling import detect_resampling
from test_resampling_accuracy import test_accuracy_against_k

# add your absolute path
casia_folder_path = r"C:\Users\chamo\Documents\Physics\Projects\Imaging and Data Processing\Automatic Fake Image Detection"
defacto_folder_path = r"C:\Users\chamo\Documents\Physics\Projects\Imaging and Data Processing\Automatic Fake Image Detection\copymove_img\img"

# for single image
#detect_resampling(suspect_image='dalle 2 real', show_images=True, fake_threshold=0.2)

# for mutiple images
#fake_image_detection(casia_folder_path=casia_folder_path, defacto_folder_path=defacto_folder_path, fake_threshold=0.2, show_bar_chart=True, show_point_plot=True,
#show_truth_table=True, num_of_images=300, produce_half_sample=True)

# for testing accuracy vs k
test_accuracy_against_k(min_k=0.18, max_k=0.23, casia_folder_path=casia_folder_path, defacto_folder_path=defacto_folder_path)
