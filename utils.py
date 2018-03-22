import os
import numpy as np

def load_mnist(path='mnist'):
    data_dir = os.path.join("./data", path)

    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)


    # X = np.concatenate((trX, teX), axis=0)
    # y = np.concatenate((trY, teY), axis=0).astype(np.int)

    # seed = 547
    # np.random.seed(seed)
    # np.random.shuffle(X)
    # np.random.seed(seed)
    # np.random.shuffle(y)

    seed = 200
    np.random.seed(seed)
    np.random.shuffle(trX)
    np.random.seed(seed)
    np.random.shuffle(trY)

    return (trX, trY, teX, teY)