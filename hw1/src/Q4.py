from math import log10
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pa
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from sklearn.cross_validation import KFold
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import PolynomialFeatures

from preprocess2 import preprocess_2


def linear_regression(data, target):
    lr = linear_model.LinearRegression(normalize=True)
    kf = KFold(len(target), n_folds=10, shuffle=True, random_state=None)
    RMSE_LINEAR = []
    for train_index, test_index in kf:
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        lr.fit(data_train, target_train)
        rmse_linear = sqrt(np.mean((lr.predict(data_test) - target_test) ** 2))
        RMSE_LINEAR.append(rmse_linear)

    F, pval = f_regression(data_test, lr.predict(data_test))
    print(pval)

    test_times = np.arange(1, 11)
    plt.figure()
    plt.plot(test_times, RMSE_LINEAR, label="RMSE in linear regression with 10-fold cv")
    #    plt.ylim(0.0, 0.12)
    # plt.title("RMSE comparison between linear regression and random forest regression")
    plt.xlabel("cross validation times")
    plt.ylabel("RMSE")
    plt.legend()

    predicted = lr.predict(data)
    index = np.arange(1, len(predicted) + 1)

    plt.figure()
    plt.scatter(index, target, s=15, color='red', label="Actual")
    plt.scatter(index, predicted, s=15, color='green', label="Fitted")
    plt.xlabel('Index')
    plt.ylabel('MEDV')
    plt.legend()

    plt.figure()
    plt.scatter(predicted, predicted - target, label="residual VS fitted values")
    plt.xlabel("fitted values")
    plt.ylabel("residual")
    plt.legend()
    #    plt.ylim(-0.8,0.4)
    plt.show()
    return RMSE_LINEAR


def polynomial_regression():
    boston = datasets.load_boston()

    # linear-regression
    lr = linear_model.LinearRegression()

    test_times = range(0, 10)

    plt.figure(1)
    # divide the data into two sets
    Train_set_x = pd.DataFrame(boston.data, index=None)
    Train_set_y = pd.DataFrame(boston.target, index=None)
    # split the data into 10 folds
    f10 = KFold(len(Train_set_x), n_folds=10, shuffle=True, random_state=None)
    # result1 linear,result2 poly-2,result3 poly-3,result4 poly-4,result5 poly-5,result6 poly-6,result7 poly-7

    for i in range(1, 8):
        results = []
        for train_index, test_index in f10:
            x_train, x_test = Train_set_x.iloc[train_index], Train_set_x.iloc[test_index]
            y_train, y_test = Train_set_y.iloc[train_index], Train_set_y.iloc[test_index]

            poly = PolynomialFeatures(degree=i)
            x_train_2 = poly.fit_transform(x_train)
            x_test_2 = poly.fit_transform(x_test)
            lr.fit(x_train_2, y_train)
            error = log10(sqrt(np.mean((lr.predict(x_test_2) - y_test) ** 2)))
            results.append(error)
        print results
        plt.plot(test_times, results, label=('RMSE of Ploynomial Regression--Degree' + str(i)))
    plt.legend(fontsize=6)
    plt.title('Comparsion of Linear Regression and Polynomial Regression--Boston')
    plt.xlabel('Test Times')
    plt.ylabel('RMSE')
    plt.ylim(0.0, 7)
    plt.show()
    plt.savefig('problem4-2')


def main():
    data = pa.read_csv("./housing_data.csv", header=None).values[:, :]
    binaryData = preprocess_2(data)
    target = data[:, 13]
    rmse_linear = linear_regression(binaryData, target)
    #rmse_l1CV=lassoCV_regression(binaryData,target)
    #print("Mean RMSE of linear regression=" + str(np.mean(rmse_linear)))
    # print(np.mean(rmse_l1CV))
    #polynomial_regression()


if __name__ == "__main__":
    main()
