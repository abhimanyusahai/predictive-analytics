import numpy as np

# Image parameters
img_dimension = [176, 208, 176]
img_dtype = np.int16

# Load/Save and basic image parameters
load_mode = False
downsample = True
blur = True
num_train_imgs = 278
num_test_imgs = 138
downsample_factor = [2, 2, 2]
blur_sigma = 2.1
data_path = "../data/"

# Voxel model parameters
voxel_model_data_path = data_path + "voxel_model/"

# HoG model parameters
hog_model_data_path = data_path + "hog_model/"
hog_model_pixels_per_cell = (16, 16)
hog_model_cells_per_block = (1, 1)

# Canny model parameters
canny_model_data_path = data_path + "canny_model/"
canny_threshold_low = 0.66
canny_threshold_high = 1.33

# Training parameters
n_splits = 5

# Model mixing parameters - choose one of 'cross-val score optimization', 'logistic regression',
# 'gbm', 'neural net'
method = 'cross-val score optimization'
