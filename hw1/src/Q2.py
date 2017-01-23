import matplotlib.pyplot as plt
import pandas as pa
from sklearn import linear_model, cross_validation
import numpy


# Get the rows from the 'selected' index.
def get_selected(all_elem, selected):
    result = []
    for i in selected:
        result.append(all_elem[i])
    return result


def Q2A(network):
    network_X = network.values[:, (0, 1, 2, 3, 4, 6)]
    network_Y = network.values[:, 5]
    model = linear_model.LinearRegression()

    regr = linear_model.LinearRegression()
    kf = cross_validation.KFold(len(network_X), 10, True)
    for train_index, test_index in kf:
        network_X_train = get_selected(network_X, train_index)
        network_Y_train = get_selected(network_Y, train_index)
        regr.fit(network_X_train, network_Y_train)

    predicted = cross_validation.cross_val_predict(model, network_X, network_Y, 10, 1, 0, None, 0)
    scores = cross_validation.cross_val_score(model, network_X, network_Y, cv=10, scoring='mean_squared_error')

    print 'All RMSEs', numpy.sqrt(-scores)
    print 'Mean RMSE', numpy.mean(numpy.sqrt(-scores))
    print 'Best RMSE', numpy.min(numpy.sqrt(-scores))
    print 'Coefficients', regr.coef_

    # Residual
    residual = []
    for i in range(len(network_X)):
        residual.append(network_Y[i] - predicted[i])

    # Plot outputs
    plt.scatter(range(len(network_X)), network_Y, color='black')
    plt.scatter(range(len(network_X)), predicted, color='blue')
    # plt.scatter(residual, predicted, color='red')

    plt.show()


def main():
    network = pa.read_csv("../Data/network_backup_dataset.csv", header=0)
    dict = {"Monday": "1", "Tuesday": "2", "Wednesday": "3", "Thursday": "4", "Friday": "5", "Saturday": "6",
            "Sunday": "7"}
    workflows = pa.unique(network["Work-Flow-ID"])
    colors = {workflows[0]: "red", workflows[1]: "orange", workflows[2]: "yellow", workflows[3]: "green",
              workflows[4]: "blue"}

    for i in dict:
        network["Day of Week"] = [s.replace(i, dict[i]) for s in network["Day of Week"]]

    network['File Name'] = [s.replace("File_", "") for s in network['File Name']]
    network['Work-Flow-ID'] = [s.replace("work_flow_", "") for s in network['Work-Flow-ID']]
    network['Day of Week'] = [int(s) for s in network['Day of Week']]
    network['File Name'] = [int(s) for s in network['File Name']]
    network['Work-Flow-ID'] = [int(s) for s in network['Work-Flow-ID']]
    Q2A(network)


if __name__ == "__main__":
    main()
