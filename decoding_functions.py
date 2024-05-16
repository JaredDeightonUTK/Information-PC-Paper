# Decoding tests

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import torch
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

def get_LOO_err(p, pos, n_bins, binary = True, plot = False):
    
    if binary:
        p[p > 0] = 1
    
    predictions = np.zeros_like(pos)
    est = KBinsDiscretizer(n_bins = n_bins, encode ='ordinal', strategy='uniform', subsample = None)
    pos_discrete = est.fit_transform(pos)

    for i in range(p.shape[0]):
        # Leave one out: use all samples except the ith for training
        p_train = np.delete(p, i, axis=0)
        pos_train = np.delete(pos_discrete, i, axis=0)

        # The ith sample is the test sample
        p_test = p[i, :].reshape(1, -1)
        
        # Train the Gaussian Naive Bayes model
        model_x = GaussianNB()
        model_y = GaussianNB()
        model_x.fit(p_train, pos_train[:,0])
        model_y.fit(p_train, pos_train[:,1])

        # Predict the position for the test sample
        pos_pred_x = model_x.predict(p_test)[0]
        pos_pred_y = model_y.predict(p_test)[0]

        # Store the prediction
        predictions[i, :] = pos_pred_x, pos_pred_y
        
    predictions_cont = est.inverse_transform(predictions)
        
    # Calculate the average error
    mse = np.mean((pos - predictions_cont)**2)
    
    if plot:
        Nx = 100
        box_width = 0.5
        box_height = 0.5

        num_traj = 2
        for i in range(0, Nx*num_traj, Nx):
            plt.figure(figsize=(5,5))
            plt.plot(pos[i:i+Nx, 0], pos[i:i+Nx, 1], label='Simulated Trajectory', c = 'Blue')
            plt.plot(predictions_cont[i:i+Nx,0], predictions_cont[i:i+Nx,1], label='Predicted Trajectory', c = 'Red')
            plt.xlim(-1*box_width/2, box_width/2)
            plt.ylim(-1*box_height/2, box_height/2)
            plt.legend()
            plt.show()
    
    return mse, predictions_cont


def get_bin_decoding_acc(p, pos, train_num = 10000, test_num = 1000, test_size = 0.2, num_axis_bins = 2, binary = True, plot = False):
    
    test_size = int(0.2*p.shape[0])

    P_train = np.float32(p[:-1*test_size])[:train_num]
    P_test = np.float32(p[test_size:])[:test_num]

    pos_train = np.float32(pos[:-1*test_size])[:train_num]
    pos_test = np.float32(pos[test_size:])[:test_num]

    P_train = np.nan_to_num((P_train - np.min(P_train, axis = 0))  / (np.max(P_train, axis = 0) - np.min(P_train, axis = 0)))

    P_train = torch.from_numpy(P_train)

    P_test = np.nan_to_num((P_test - np.min(P_test, axis = 0))  / (np.max(P_test, axis = 0) - np.min(P_test, axis = 0)))
    P_test = torch.from_numpy(P_test)

    # Make positive just 1 or add noise
    
    if binary:
        P_train[P_train > 0] = 1
        P_test[P_test > 0] = 1


    #P_train = P_train + torch.rand(size = P_train.size())

    num_bins_per_axis = num_axis_bins  # Define the number of bins per axis

    est_x = KBinsDiscretizer(n_bins=num_bins_per_axis, encode='ordinal', strategy='uniform', subsample = None)
    est_y = KBinsDiscretizer(n_bins=num_bins_per_axis, encode='ordinal', strategy='uniform', subsample = None)

    pos_x_binned_train = est_x.fit_transform(pos_train[:, 0:1]).flatten()  # Bins for x-axis
    pos_y_binned_train = est_y.fit_transform(pos_train[:, 1:2]).flatten()  # Bins for y-axis

    pos_x_binned_test = est_x.fit_transform(pos_test[:, 0:1]).flatten()  # Bins for x-axis
    pos_y_binned_test = est_y.fit_transform(pos_test[:, 1:2]).flatten()  # Bins for y-axis

    pos_train_binned = pos_x_binned_train * num_bins_per_axis + pos_y_binned_train
    pos_train_binned = torch.from_numpy(pos_train_binned).long()

    pos_test_binned = pos_x_binned_test * num_bins_per_axis + pos_y_binned_test
    pos_test_binned = torch.from_numpy(pos_test_binned).long()

    clf = svm.SVC(kernel = 'linear')

    clf.fit(P_train, pos_train_binned)

    train_predictions = clf.predict(P_train)
    test_predictions = clf.predict(P_test)

    pos_train_binned = np.array(pos_train_binned)
    pos_test_binned = np.array(pos_test_binned)
    
    train_acc = sum(train_predictions == pos_train_binned)/pos_train_binned.shape[0]
    test_acc = sum(test_predictions == pos_test_binned)/pos_test_binned.shape[0]
    
    if plot:
        plt.figure()
        plt.scatter(pos_train[:,0], pos_train[:,1], c = (train_predictions == pos_train_binned))
        plt.show()
    
    
        plt.figure()
        plt.scatter(pos_test[:,0], pos_test[:,1], c = (test_predictions == pos_test_binned))
        plt.show()
    
    return test_acc, test_predictions, pos_test, pos_test_binned