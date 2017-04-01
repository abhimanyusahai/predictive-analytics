import config
import load
import models
import numpy as np
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import code


def main():
    # Load data
    if (config.load_mode):
        load.load_data(config.data_path + "set_train/", config.data_path + "set_test/")

    # Create the models
    model_list = []
    num_models = 3
    # model_list.append(models.VoxelModel(3, 0.01, downsample_factor = [2, 2, 2],
    #                                     blur_sigma = 2.1, pca = False, model = 'gbm'))
    # model_list.append(models.HoGModel(0.01, model='svm', blur_sigma = 1.5))
    model_list.append(models.CannyModel(0.01, blur_sigma=1, canny_sigma=1, model='svm', num_cubes=27))

    # Fit the models to training data
    train_targets = np.genfromtxt(config.data_path + "targets.csv", delimiter=',')[:,2]
    cv_predictions = []
    training_predictions = []
    for i in range(0, len(model_list)):
        model_list[i].fit(train_targets)
        cv_predictions.extend(model_list[i].cv_predictions)
        training_predictions.extend(model_list[i].train_predictions)

    # Create predictions based on each model
    model_predictions = []
    for i in range(0, len(model_list)):
        model_list[i].predict()
        model_predictions.extend(model_list[i].predictions)

    # Figure out optimal model mixing method and calculate weighted predictions
    if config.method == "cross-val score optimization":
        print "Optimizing model mixing using cross-val score optimization method..."
        def loss(weights):
            predictions = np.array(cv_predictions[0])
            for i in range(1, num_models):
                predictions = np.c_[predictions, weights[i - 1]*(cv_predictions[i] - cv_predictions[0])]
            predictions = np.sum(predictions, axis=1)
            return log_loss(train_targets, predictions)
        initial_weights = np.zeros(((num_models - 1),))
        optimal_weights = minimize(loss, initial_weights, method='L-BFGS-B', bounds=[(0, None)]*(num_models - 1)).x
        optimal_weights = np.insert(optimal_weights, 0, (1 - np.sum(optimal_weights)))
        print optimal_weights
        final_predictions = np.array(optimal_weights[0]*model_predictions[0])
        for i in range(1, num_models):
            final_predictions = np.c_[final_predictions, optimal_weights[i]*model_predictions[i]]
        final_predictions = np.sum(final_predictions, axis=1)
    elif config.method == "gbm":
        print "Optimizing model mixing with GBM..."
        train_crossval_predictions = np.transpose(np.array(cv_predictions))
        gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1)
        gbm.fit(train_crossval_predictions, train_targets)
        train_predictions = gbm.predict_proba(train_crossval_predictions)[:,1]
        test_data = np.transpose(np.array(model_predictions))
        final_predictions = gbm.predict_proba(test_data)[:,1]

    # Write weighted predictions to file
    model_predictions = np.transpose(np.array(model_predictions))
    fp = open(config.data_path + "predictions.csv", 'w')
    fp.write("ID,Prediction\n")
    for i in range(0,config.num_test_imgs):
        fp.write(str(i+1)+","+str(final_predictions[i])+"\n")
    fp.close()


if __name__ == "__main__":
    main()
