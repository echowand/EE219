from math import sqrt
import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
from sklearn import linear_model, cross_validation
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer


# Compare the linear regression model and random forest regression model
def linear_and_random_forest_regression(data, target, network):
    lr = linear_model.LinearRegression(normalize=True)
    rfr = RandomForestRegressor(n_estimators=30, max_depth=12, max_features='auto')
    kf = KFold(len(target), n_folds=10, shuffle=True, random_state=None)
    RMSE_LINEAR = []
    RMSE_RFR = []
    for train_index, test_index in kf:
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        lr.fit(data_train, target_train)
        rfr = rfr.fit(data_train, target_train)
        rmse_linear = sqrt(np.mean((lr.predict(data_test) - target_test) ** 2))
        RMSE_LINEAR.append(rmse_linear)
        rmse_rfr = sqrt(np.mean((rfr.predict(data_test) - target_test) ** 2))
        RMSE_RFR.append(rmse_rfr)

    F, pval = f_regression(data_test, lr.predict(data_test))
    print(np.mean(RMSE_RFR))
    print(pval)
    test_times = np.arange(1, 11)
    plt.figure()
    plt.plot(test_times, RMSE_LINEAR, label="RMSE in linear regression with 10-fold cv")
    plt.plot(test_times, RMSE_RFR, label="RMSE in random forest regression with 10-fold cv")
    plt.ylim(0.0, 0.12)
    plt.title("linear regression and random forest regression RMSE")
    plt.xlabel("cross validation #")
    plt.ylabel("RMSE")
    plt.legend()

    network['predicted_lr'] = lr.predict(data)
    network['predicted_rfr'] = rfr.predict(data)
    network_time_target = network.groupby(["Week #", "Day of Week", "Backup Start Time - Hour of Day"])[
        "Size of Backup (GB)"].sum()
    network_time_predict_lr = network.groupby(["Week #", "Day of Week", "Backup Start Time - Hour of Day"])[
        "predicted_lr"].sum()
    network_time_predict_rfr = network.groupby(["Week #", "Day of Week", "Backup Start Time - Hour of Day"])[
        "predicted_rfr"].sum()
    time = np.arange(1, len(network_time_target) + 1)

    plt.figure()
    plt.scatter(time, network_time_target, s=15, color='red', label="Actual values over time")
    plt.scatter(time, network_time_predict_lr, s=15, color='green', label="predicted values with linear model")
    plt.xlabel('Time')
    plt.ylabel('Size of backup(GB)')
    plt.ylim(-2, 12)
    plt.legend()

    plt.figure()
    plt.plot(time[0:120], network_time_predict_rfr[0:120], label="predicted values with random forest tree model")
    plt.legend()

    plt.figure()
    plt.scatter(lr.predict(data), lr.predict(data) - target, label="residual VS fitted values")
    plt.xlabel("fitted values")
    plt.ylabel("residual")
    plt.legend()
    plt.ylim(-0.8, 0.4)
    plt.show()
    return RMSE_LINEAR


# Tuning the parameter of randomforest model.
def random_forest_tuning_parameters(data, target, network):
    kf = KFold(len(target), 10, shuffle=True);
    RMSE_BEST = 10
    rfr_best = RandomForestRegressor(n_estimators=30, max_features=len(data[0]), max_depth=8)
    for nEstimators in range(29, 31, 1):
        for maxFeatures in range(len(data[0]) - 1, len(data[0] + 1)):
            for maxDepth in range(11, 13, 1):
                rfr = RandomForestRegressor(n_estimators=nEstimators, max_features=maxFeatures, max_depth=maxDepth)
                RMSE_RFR = []
                for train_index, test_index in kf:
                    data_train, data_test = data[train_index], data[test_index]
                    target_train, target_test = target[train_index], target[test_index]
                    rfr.fit(data_train, target_train)
                    rmse_rfr = sqrt(np.mean((rfr.predict(data_test) - target_test) ** 2))
                    RMSE_RFR.append(rmse_rfr)
                if RMSE_BEST > np.mean(RMSE_RFR):
                    rfr_best = rfr
                    RMSE_BEST = np.mean(RMSE_RFR)
    kf_final = KFold(len(target), 10, shuffle=True);
    RMSE_FINAL = []
    for train_index, test_index in kf_final:
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        rfr_best.fit(data_train, target_train)
        rmse_rfr = sqrt(np.mean((rfr_best.predict(data_test) - target_test) ** 2))
        RMSE_FINAL.append(rmse_rfr)
    plt.figure()
    plt.plot(range(1, len(RMSE_FINAL) + 1), RMSE_FINAL)
    plt.title("The best RMSE with random forest")
    plt.xlabel("cross validation times")
    plt.ylabel("RMSE")
    plt.show()
    print(np.mean(RMSE_FINAL))
    return RMSE_FINAL


# Fit the data with neural network
def neural_network(data, target):
    DS = SupervisedDataSet(len(data[0]), 1)
    nn = buildNetwork(len(data[0]), 7, 1, bias=True)
    kf = KFold(len(target), 10, shuffle=True);
    RMSE_NN = []
    for train_index, test_index in kf:
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        for d, t in zip(data_train, target_train):
            DS.addSample(d, t)
        bpTrain = BackpropTrainer(nn, DS, verbose=True)
        # bpTrain.train()
        bpTrain.trainUntilConvergence(maxEpochs=10)
        p = []
        for d_test in data_test:
            p.append(nn.activate(d_test))

        rmse_nn = sqrt(np.mean((p - target_test) ** 2))
        RMSE_NN.append(rmse_nn)
        DS.clear()
    time = range(1, 11)
    plt.figure()
    plt.plot(time, RMSE_NN)
    plt.xlabel('cross-validation time')
    plt.ylabel('RMSE')
    plt.show()
    print(np.mean(RMSE_NN))


def main():
    network = pa.read_csv("./network_backup_dataset.csv", header=0)
    dict = {"Monday": "1", "Tuesday": "2", "Wednesday": "3", "Thursday": "4", "Friday": "5", "Saturday": "6",
            "Sunday": "7"}
    for i in dict:
        network['Day of Week'] = [s.replace(i, dict[i]) for s in network['Day of Week']]

    network['File Name'] = [s.replace("File_", "") for s in network['File Name']]
    network['Work-Flow-ID'] = [s.replace("work_flow_", "") for s in network['Work-Flow-ID']]
    network['Day of Week'] = [int(s) - 1 for s in network['Day of Week']]
    network['File Name'] = [int(s) for s in network['File Name']]
    network['Work-Flow-ID'] = [int(s) for s in network['Work-Flow-ID']]

    data1 = network.values[:, 0:5]
    data2 = network.values[:, 6:7]
    data = np.concatenate((data1, data2), axis=1)
    target = network.values[:, 5]

    # linear and random forest model
    linear_and_random_forest_regression(data, target, network)

    # tuning the parameters of the random forest
    random_forest_tuning_parameters(data, target, network)

    # neural network
    neural_network(data, target)


if __name__ == "__main__":
    main()
