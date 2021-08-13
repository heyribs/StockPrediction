import numpy as np

def split_data(data):
    shape = data.shape
    # print(data.shape) # (850, 2, 5, 37)
    data_former = []
    data_latter = []
    for i in range(shape[0]):
        data_former.append(data[i, 0, :, :])
        data_latter.append(data[i, 1, :, :])
    data_former = np.array(data_former, dtype=float)
    data_latter = np.array(data_latter, dtype=float)
    return data_former, data_latter