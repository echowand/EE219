import pandas as pd
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


# load u.data
def load_data():
    df = pd.read_csv('ml-100k/u.data', delimiter='\t', names=['user_id', 'item_id', 'rating', 'timestamp'], header=0)
    # R: users on rows and movies on columns: 943*1683
    R = df.pivot_table(index=['user_id'], columns=['item_id'], values='rating', fill_value=0).values
    # W: weight matrix
    W = R.copy()
    W[W > 0] = 1
    return R, W


# calculate squared error
def squared_error(R, W, U, V):
    return np.sum((W * (R - np.dot(U, V))) ** 2)


# default iteration = 100
def matrix_factorize(R, W, k, reg_param=0, num_iterations=100):
    eps = 1e-5

    m, n = R.shape
    U = np.maximum(eps, 5 * np.random.rand(m, k))
    V = np.maximum(eps, linalg.lstsq(U, R)[0])

    WR = W * R

    for i in range(num_iterations):
        # V as a constraint
        top = np.dot(WR, V.T)
        bottom = np.add(
            np.dot(W * np.dot(U, V), V.T),
            reg_param * U
        ) + eps
        U = np.maximum(eps, U * top / bottom)

        # U as a constraint
        top = np.dot(U.T, WR)
        bottom = np.add(
            np.dot(U.T, W * np.dot(U, V)),
            reg_param * V
        ) + eps
        V = np.maximum(eps, V * top / bottom)

    print "Matrix Factorization MSQE: ", squared_error(R, W, U, V)

    return U, V


def draw_ROC(predicted, actual, k, alpha):
    tp = 0  # true positive
    fp = 0  # false positive
    fn = 0  # false negative

    threshold_value = np.arange(1, 6, 1)
    precision = np.zeros(len(threshold_value))
    recall = np.zeros(len(threshold_value))

    for x, t in enumerate(threshold_value):
        tp = np.sum(actual[predicted >= t] >= t)
        fp = np.sum(actual[predicted >= t] < t)
        fn = np.sum(actual[predicted < t] >= t)

        # https://en.wikipedia.org/wiki/Precision_and_recall
        precision[x] = tp / float(tp + fp)
        recall[x] = tp / float(tp + fn)

    plt.figure(1)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('ROC k={0} lambda={1}'.format(k, alpha))
    plt.scatter(precision, recall, s=60, marker='o')
    plt.plot(precision, recall)
    plt.savefig("pics/ROC_k=" + str(k) + "_l=" + str(alpha) + ".png", format='png')
    plt.clf()


if __name__ == "__main__":
    R, W = load_data()

    klist = [10, 50, 100]
    lambdalist = [0.01, 0.1, 1]

    for k in klist:
        print "k = " + str(k) + ", iteration = 200"
        U, V = matrix_factorize(W, R, k, reg_param=0, num_iterations=200)

    for k in klist:
        for l in lambdalist:
            print "k = " + str(k) + ", lambda = " + str(l)
            U, V = matrix_factorize(R, W, k, reg_param=l)
            draw_ROC(np.dot(U, V), R, k, l)
