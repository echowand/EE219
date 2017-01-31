import numpy as np

#preprocess data
def preprocess_2(data):
    rowNO = len(data)
    indices = data[:, 8:9]
    bindices = np.zeros((rowNO, 9))
    i = 0
    for tmp in np.nditer(indices):
        if (tmp == 24):
            bindices[i, 8] = 1
        else:
            bindices[i, int(tmp) - 1] = 1
        i += 1

    preprocessed = np.concatenate((data[:, 0:8], bindices, data[:, 9:13]), axis=1)
    return preprocessed



