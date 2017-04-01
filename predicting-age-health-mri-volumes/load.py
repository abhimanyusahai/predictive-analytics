import numpy as np
import config
import nibabel as nib
from scipy.ndimage.filters import gaussian_filter

def load_data(train_data_path, test_data_path):

    # Helper functions

    def downsample(img, factor):
        return img[::factor[0], ::factor[1], ::factor[2]]

    def gaussian_blur(img, sigma):
        return gaussian_filter(img, sigma)

    def remove_zeros(img_array, zero_cols=None):
        if (not zero_cols):
            zero_cols = np.where(~np.any(img_array, axis=0))
        return np.delete(img_array, zero_cols, axis=1), zero_cols

    # Load the train data
    all_imgs = []
    for i in xrange(0, config.num_train_imgs):
        print "Loading and pre-processing train image " + str(i + 1) + "..."
        img = nib.load(train_data_path + "train_" + str(i + 1) + ".nii").get_data()
        if (config.downsample):
            img = downsample(img, config.downsample_factor)
        if (config.blur):
            img = gaussian_blur(img, config.blur_sigma)
        all_imgs.append(img.reshape(img.size))

    # Remove voxels that are zero in all images and save data
    print "All images loaded, removing zeros voxels..."
    all_imgs, train_zero_cols = remove_zeros(np.array(all_imgs))
    np.save(config.data_path + "loaded_train_img_data.npy", all_imgs)

    # Load the test data
    all_imgs = []
    for i in xrange(0, config.num_test_imgs):
        print "Loading and pre-processing test image " + str(i + 1) + "..."
        img = nib.load(test_data_path + "test_" + str(i + 1) + ".nii").get_data()
        if (config.downsample):
            img = downsample(img, config.downsample_factor)
        if (config.blur):
            img = gaussian_blur(img, config.blur_sigma)
        all_imgs.append(img.reshape(img.size))

    # Remove voxels that are zero in all images and save data
    print "All images loaded, removing zeros voxels..."
    all_imgs = remove_zeros(np.array(all_imgs), train_zero_cols)[0]
    np.save(config.data_path + "loaded_test_img_data.npy", all_imgs)
