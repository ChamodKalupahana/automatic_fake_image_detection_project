import numpy as np
import matplotlib.pyplot as plt
from loading_data import load_image

orginal_image = load_image('goosefair')

plt.imshow(orginal_image)
plt.show()