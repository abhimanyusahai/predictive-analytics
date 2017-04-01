import config
import os
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from scipy.ndimage.filters import gaussian_filter
import nibabel as nib
from skimage.feature import hog
from skimage.feature import canny
import code
from copy import deepcopy
from sklearn.ensemble import GradientBoostingClassifier


class VoxelModel(object):

    def __init__(self, num_segments, regularization, model = 'logistic',
                    downsample_factor = [1, 1, 1], blur_sigma = 0, pca = False):
        self.num_segments = num_segments
        self.regularization = regularization
        if model == 'logistic': self.base_model = LogisticRegression(C = regularization)
        elif model == 'svm': self.base_model = SVC(C = regularization, probability = True)
        elif model == 'gbm': self.base_model = GradientBoostingClassifier(n_estimators=100,
                                                                      learning_rate=0.1, max_depth=8)
        self.downsample_factor = downsample_factor
        self.blur_sigma = blur_sigma
        if (pca): self.pca = []
        self.models = []
        self.cv_predictions = []
        self.train_predictions = []
        self.predictions = []
        print("Initialized a voxel model with %d segments..." % self.num_segments)


    def data_cached(self):
        if os.path.exists(config.voxel_model_data_path +
                        "voxel_model_train_data_seg1.npy"): return True
        else: return False


    def model_cached(self):
        if os.path.exists(config.voxel_model_data_path +
                            "voxelmodelseg1"): return True
        else: return False


    def load_data(self, seg):
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
            print "Loading train image " + str(i + 1) + "..."
            img = nib.load(config.voxel_model_data_path + "set_train/c" +
                           str(seg + 1) + "train_" + str(i + 1) + ".nii").get_data()
            img = downsample(img, self.downsample_factor)
            img = gaussian_blur(img, self.blur_sigma)
            all_imgs.append(img.reshape(img.size))

        # Remove voxels that are zero in all images and save data
        print "All images loaded, removing zeros voxels..."
        all_imgs, train_zero_cols = remove_zeros(np.array(all_imgs))
        np.save(config.voxel_model_data_path + "voxel_model_train_data_seg"
                + str(seg + 1) + ".npy", all_imgs)

        # Load the test data
        all_imgs = []
        for i in xrange(0, config.num_test_imgs):
            print "Loading test image " + str(i + 1) + "..."
            img = nib.load(config.voxel_model_data_path + "set_test/c" +
                           str(seg + 1) + "test_" + str(i + 1) + ".nii").get_data()
            img = downsample(img, self.downsample_factor)
            img = gaussian_blur(img, self.blur_sigma)
            all_imgs.append(img.reshape(img.size))

        # Remove voxels that are zero in all images and save data
        print "All images loaded, removing zeros voxels..."
        all_imgs = remove_zeros(np.array(all_imgs), train_zero_cols)[0]
        np.save(config.voxel_model_data_path + "voxel_model_test_data_seg"
                + str(seg + 1) + ".npy", all_imgs)


    def fit(self, train_targets):
        # Load the data if not already cached
        if not self.data_cached():
            for seg in range(0, self.num_segments):
                print("Loading data for segment %d..." % (seg + 1))
                self.load_data(seg)

        # Use saved model object if cached
        if self.model_cached():
            for seg in range(0, self.num_segments):
                model = joblib.load(config.voxel_model_data_path + "voxelmodelseg"
                                    + str(seg + 1))
                self.models.append(model)
                if hasattr(self, 'pca'):
                    pca = joblib.load(config.voxel_model_data_path + "pcaseg"
                                        + str(seg + 1))
                    self.pca.append(pca)
            self.cv_predictions = np.load(config.voxel_model_data_path +
                                          "voxel_model_cv_predictions.npy")
            self.train_predictions = np.load(config.voxel_model_data_path +
                                             "voxel_model_train_predictions.npy")
        else:
            # Extract features and pre-process
            for seg in range(0, self.num_segments):
                # Load basic features
                print("Fitting voxel model for segment %d..." % (seg + 1))
                train_data = np.load(config.voxel_model_data_path +
                                     "voxel_model_train_data_seg" +
                                     str(seg + 1) + ".npy")

                # Perform PCA if applicable
                if hasattr(self, 'pca'):
                    print "Performing PCA..."
                    pca = PCA(config.num_train_imgs)
                    train_data = pca.fit_transform(train_data)
                    joblib.dump(pca, config.voxel_model_data_path + "pcaseg"
                                        + str(seg + 1))
                    self.pca.append(pca)

                # Train the data and perform cross-validation
                print "Training model..."
                model = self.base_model
                kf = KFold(n_splits=config.n_splits)
                cv_score = 0
                cv_predictions = []
                for train, test in kf.split(train_data):
                    model.fit(train_data[train], train_targets[train])
                    predictions = model.predict_proba(train_data[test])
                    score = log_loss(train_targets[test], predictions[:,1])
                    cv_predictions.extend(predictions[:,1])
                    cv_score += score
                cv_score = cv_score/config.n_splits
                print "Cross validation log loss is: " + str(cv_score)
                self.cv_predictions.append(cv_predictions)
                model.fit(train_data, train_targets)
                self.train_predictions.append(model.predict_proba(train_data)[:,1])
                joblib.dump(model, config.voxel_model_data_path + "voxelmodelseg"
                            + str(seg + 1))
                self.models.append(deepcopy(model))
            np.save(config.voxel_model_data_path + "voxel_model_cv_predictions",
                    self.cv_predictions)
            np.save(config.voxel_model_data_path + "voxel_model_train_predictions",
                    self.train_predictions)


    def predict(self):
        # Extract features and pre-process
        for seg in range(0, self.num_segments):
            # Load basic features
            print("Predicting based on voxel model for segment %d..." % (seg + 1))
            test_data = np.load(config.voxel_model_data_path +
                                 "voxel_model_test_data_seg" +
                                 str(seg + 1) + ".npy")

            # Perform PCA if applicable
            if hasattr(self, 'pca'):
                print "Applying PCA transform on test data..."
                test_data = self.pca[seg].transform(test_data)

            # Predict
            predictions = self.models[seg].predict_proba(test_data)
            self.predictions.append(predictions[:,1])


class HoGModel(object):

    def __init__(self, regularization, model = 'logistic',
                 downsample_factor = [1, 1, 1], blur_sigma = 0):
        self.regularization = regularization
        if model == 'logistic': self.base_model = LogisticRegression(C = regularization)
        elif model == 'svm': self.base_model = SVC(C = regularization, probability = True, gamma=0.0000001)
        self.downsample_factor = downsample_factor
        self.blur_sigma = blur_sigma
        self.models = []
        self.cv_predictions = []
        self.predictions = []
        self.train_predictions = []
        print("Initialized a HoG model...")


    def data_cached(self):
        if os.path.exists(config.hog_model_data_path +
                        "hog_model_train_data.npy"): return True
        else: return False


    def model_cached(self):
        if os.path.exists(config.hog_model_data_path +
                            "hogmodelplane1"): return True
        else: return False


    def load_data(self):
        # Helper functions

        def downsample(img, factor):
            return img[::factor[0], ::factor[1], ::factor[2]]

        def gaussian_blur(img, sigma):
            return gaussian_filter(img, sigma)

        def compute_hog(img3d):
            result = [[], [], []]
            for i in range(0, 3):
                    for j in range(0, config.img_dimension[i]/self.downsample_factor[i]):
                        if i == 0: img = img3d[j, :, :]
                        elif i == 1: img = img3d[:, j, :]
                        else: img = img3d[:, :, j]
                        features = hog(img,
                                       pixels_per_cell=config.hog_model_pixels_per_cell,
                                       cells_per_block=config.hog_model_cells_per_block)
                        result[i].extend(features)
            return result

        # Load and save train data
        all_imgs = [[], [], []]
        for i in xrange(0, config.num_train_imgs):
            print "Loading train image " + str(i + 1) + "..."
            img = nib.load(config.data_path + "set_train/train_" +
                           str(i + 1) + ".nii").get_data()
            img = img.reshape(*config.img_dimension)
            img = downsample(img, self.downsample_factor)
            img = gaussian_blur(img, self.blur_sigma)
            img_hog = compute_hog(img)
            for i in range(0, 3): all_imgs[i].append(img_hog[i])
        for i in range(0, 3): all_imgs[i] = np.array(all_imgs[i])
        np.save(config.hog_model_data_path + "hog_model_train_data.npy",
                np.array(all_imgs))

        # Load the save test data
        all_imgs = [[], [], []]
        for i in xrange(0, config.num_test_imgs):
            print "Loading test image " + str(i + 1) + "..."
            img = nib.load(config.data_path + "set_test/test_" +
                           str(i + 1) + ".nii").get_data()
            img = img.reshape(*config.img_dimension)
            img = downsample(img, self.downsample_factor)
            img = gaussian_blur(img, self.blur_sigma)
            img_hog = compute_hog(img)
            for i in range(0, 3): all_imgs[i].append(img_hog[i])
        for i in range(0, 3): all_imgs[i] = np.array(all_imgs[i])
        np.save(config.hog_model_data_path + "hog_model_test_data.npy",
                np.array(all_imgs))


    def fit(self, train_targets):
        # Load and extract features if not cached
        if not self.data_cached():
            self.load_data()

        # Use saved model object if cached
        if self.model_cached():
            for i in range(0, 3):
                model = joblib.load(config.hog_model_data_path + "hogmodelplane"
                                    + str(i + 1))
                self.models.append(model)
            self.cv_predictions = np.load(config.hog_model_data_path +
                                          "hog_model_cv_predictions.npy")
            self.train_predictions = np.load(config.hog_model_data_path +
                                             "hog_model_train_predictions.npy")
        else:
            # Train the data and perform cross-validation
            gradient_data = np.load(config.hog_model_data_path +
                                 "hog_model_train_data.npy")
            for i in range(0, 3):
                print "Training model for gradients in plane %d" % (i + 1)
                model = self.base_model
                train_data = gradient_data[i]
                kf = KFold(n_splits=config.n_splits)
                cv_score = 0
                cv_predictions = []
                for train, test in kf.split(train_data):
                    model.fit(train_data[train], train_targets[train])
                    predictions = model.predict_proba(train_data[test])
                    score = log_loss(train_targets[test], predictions[:,1])
                    cv_predictions.extend(predictions[:,1])
                    cv_score += score
                cv_score = cv_score/config.n_splits
                print "Cross validation log loss is: " + str(cv_score)
                self.cv_predictions.append(cv_predictions)
                model.fit(train_data, train_targets)
                self.train_predictions.append(model.predict_proba(train_data)[:,1])
                joblib.dump(model, config.hog_model_data_path + "hogmodelplane"
                            + str(i + 1))
                self.models.append(deepcopy(model))
            np.save(config.hog_model_data_path + "hog_model_cv_predictions",
                    self.cv_predictions)
            np.save(config.hog_model_data_path + "hog_model_train_predictions",
                    self.train_predictions)


    def predict(self):
        # Extract features and pre-process
        gradient_data = np.load(config.hog_model_data_path +
                             "hog_model_test_data.npy")
        for i in range(0, 3):
            # Load basic features
            print("Predicting based on hog model for plane %d..." % (i + 1))
            test_data = gradient_data[i]
            predictions = self.models[i].predict_proba(test_data)
            self.predictions.append(predictions[:,1])


class CannyModel(object):

    def __init__(self, regularization, model = 'logistic',
                 downsample_factor = [1, 1, 1], blur_sigma=0, canny_sigma=0, num_cubes=27):
        self.regularization = regularization
        if model == 'logistic': self.base_model = LogisticRegression(C = regularization)
        elif model == 'svm': self.base_model = SVC(C = regularization, probability = True, gamma=.0000001)
        self.downsample_factor = downsample_factor
        self.blur_sigma = blur_sigma
        self.canny_sigma = canny_sigma
        self.num_cubes = num_cubes
        self.models = []
        self.cv_predictions = []
        self.train_predictions = []
        self.predictions = []
        print("Initialized a Canny model...")

    def data_cached(self):
        if os.path.exists(config.canny_model_data_path +
                        "canny_model_train_data.npy"): return True
        else: return False


    def model_cached(self):
        if os.path.exists(config.canny_model_data_path +
                            "cannymodelplane1"): return True
        else: return False


    def load_data(self):
        # Helper functions

        def downsample(img, factor):
            return img[::factor[0], ::factor[1], ::factor[2]]

        def gaussian_blur(img, sigma):
            return gaussian_filter(img, sigma)

        def compute_canny_features3d(img3d):
            result = [[], [], []]
            splits_per_side = int(self.num_cubes ** (1.0/3))
            splits_per_image = int(self.num_cubes/splits_per_side)
            for i in range(0, 3):
                features = np.zeros(((config.img_dimension[i]/self.downsample_factor[i]),
                                     splits_per_image))
                for j in range(0, config.img_dimension[i]/self.downsample_factor[i]):
                    if i == 0:
                        img = img3d[j, :, :]
                        dim1 = config.img_dimension[1]/self.downsample_factor[1]
                        dim2 = config.img_dimension[2]/self.downsample_factor[2]
                    elif i == 1:
                        img = img3d[:, j, :]
                        dim1 = config.img_dimension[0]/self.downsample_factor[0]
                        dim2 = config.img_dimension[2]/self.downsample_factor[2]
                    else:
                        img = img3d[:, :, j]
                        dim1 = config.img_dimension[0]/self.downsample_factor[0]
                        dim2 = config.img_dimension[1]/self.downsample_factor[1]
                    canny_img = canny(img, self.canny_sigma, config.canny_threshold_low*np.mean(img),
                                      config.canny_threshold_high*np.mean(img))
                    for k in xrange(0, splits_per_side):
                        for l in xrange(0, splits_per_side):
                            subimg = canny_img[(k*dim1/splits_per_side):
                                               ((k+1)*dim1/splits_per_side),
                                               (l*dim2/splits_per_side):
                                               ((l+1)*dim2/splits_per_side)]
                            num_edge_pixels = np.count_nonzero(subimg)
                            features[j, k*splits_per_side+l] = num_edge_pixels
                features = np.array_split(features, splits_per_side)
                for j in xrange(0, splits_per_side):
                    result[i].extend(np.sum(features[j], axis=0))
            return result

        # Load and save train data
        all_imgs = [[], [], []]
        for i in xrange(0, config.num_train_imgs):
            print "Loading train image " + str(i + 1) + "..."
            img = nib.load(config.data_path + "set_train/train_" +
                           str(i + 1) + ".nii").get_data()
            img = img.reshape(*config.img_dimension)
            img = downsample(img, self.downsample_factor)
            img = gaussian_blur(img, self.blur_sigma)
            canny_features = compute_canny_features3d(img)
            for i in range(0, 3): all_imgs[i].append(canny_features[i])
        for i in range(0, 3): all_imgs[i] = np.array(all_imgs[i])
        np.save(config.canny_model_data_path + "canny_model_train_data.npy",
                np.array(all_imgs))

        # Load the save test data
        all_imgs = [[], [], []]
        for i in xrange(0, config.num_test_imgs):
            print "Loading test image " + str(i + 1) + "..."
            img = nib.load(config.data_path + "set_test/test_" +
                           str(i + 1) + ".nii").get_data()
            img = img.reshape(*config.img_dimension)
            img = downsample(img, self.downsample_factor)
            img = gaussian_blur(img, self.blur_sigma)
            canny_features = compute_canny_features3d(img)
            for i in range(0, 3): all_imgs[i].append(canny_features[i])
        for i in range(0, 3): all_imgs[i] = np.array(all_imgs[i])
        np.save(config.canny_model_data_path + "canny_model_test_data.npy",
                np.array(all_imgs))


    def fit(self, train_targets):
        # Load and extract features if not cached
        if not self.data_cached():
            self.load_data()

        # Use saved model object if cached
        if self.model_cached():
            for i in range(0, 3):
                model = joblib.load(config.canny_model_data_path + "cannymodelplane"
                                    + str(i + 1))
                self.models.append(model)
            self.cv_predictions = np.load(config.canny_model_data_path +
                                          "canny_model_cv_predictions.npy")
            self.train_predictions = np.load(config.canny_model_data_path +
                                             "canny_model_train_predictions.npy")
        else:
            # Train the data and perform cross-validation
            canny_data = np.load(config.canny_model_data_path +
                                 "canny_model_train_data.npy")
            random = [0, 0, 0]
            for i in range(0, 3):
                print "Training model for edges in plane %d" % (i + 1)
                model = self.base_model
                train_data = canny_data[i]
                kf = KFold(n_splits=config.n_splits)
                cv_score = 0
                cv_predictions = []
                for train, test in kf.split(train_data):
                    model.fit(train_data[train], train_targets[train])
                    predictions = model.predict_proba(train_data[test])
                    score = log_loss(train_targets[test], predictions[:,1])
                    cv_predictions.extend(predictions[:,1])
                    cv_score += score
                cv_score = cv_score/config.n_splits
                print "Cross validation log loss is: " + str(cv_score)
                self.cv_predictions.append(cv_predictions)
                model.fit(train_data, train_targets)
                self.train_predictions.append(model.predict_proba(train_data)[:,1])
                joblib.dump(model, config.canny_model_data_path + "cannymodelplane"
                            + str(i + 1))
                self.models.append(deepcopy(model))
            self.cv_predictions = np.array(self.cv_predictions)
            np.save(config.canny_model_data_path + "canny_model_cv_predictions",
                    self.cv_predictions)
            np.save(config.canny_model_data_path + "canny_model_train_predictions",
                    self.train_predictions)


    def predict(self):
        # Extract features and pre-process
        canny_data = np.load(config.canny_model_data_path + "canny_model_test_data.npy")
        for i in range(0, 3):
            # Load basic features
            print("Predicting based on canny model for plane %d..." % (i + 1))
            test_data = canny_data[i]
            predictions = self.models[i].predict_proba(test_data)
            self.predictions.append(predictions[:,1])
