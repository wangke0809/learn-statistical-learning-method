from utils import *
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification

def test_mnist():
    trX, trY, teX, teY = load_mnist()

    # get 2 class data and label
    train_datas = []
    train_labels = []
    test_datas = []
    test_labels = []

    for x in range(trX.shape[0]):
        if trY[x] == 1.0 or trY[x]== 8.0:
            train_datas.append(trX[x].flatten())
            train_labels.append(trY[x])

    for x in range(teX.shape[0]):
        if teY[x] == 1.0 or teY[x]== 8.0:
            test_datas.append(trX[x].flatten())
            test_labels.append(trY[x])

    print(np.array(train_datas).shape)

    clf = Perceptron(penalty='l2', fit_intercept=False ,max_iter=500, shuffle=False)
    clf.fit(np.array(train_datas),np.array(train_labels))

    print(clf.coef_)
    print(clf.intercept_)

    acc = clf.score(np.array(test_datas),np.array(test_labels))
    print(acc)

def test():
    x,y = make_classification(n_samples=1000, n_features=2,n_redundant=0,n_informative=1,n_clusters_per_class=1)
    #训练数据和测试数据
    x_data_train = x[:800,:]
    x_data_test = x[800:,:]
    y_data_train = y[:800]
    y_data_test = y[800:]

    #正例和反例
    positive_x1 = [x[i,0] for i in range(1000) if y[i] == 1]
    positive_x2 = [x[i,1] for i in range(1000) if y[i] == 1]
    negetive_x1 = [x[i,0] for i in range(1000) if y[i] == 0]
    negetive_x2 = [x[i,1] for i in range(1000) if y[i] == 0]

    from sklearn.linear_model import Perceptron
    #定义感知机
    clf = Perceptron(fit_intercept=False, max_iter=3000, shuffle=False)
    #使用训练数据进行训练
    clf.fit(x_data_train,y_data_train)
    #得到训练结果，权重矩阵
    print(clf.coef_)
    #超平面的截距
    print(clf.intercept_)
    #利用测试数据进行验证
    acc = clf.score(x_data_test,y_data_test)
    print(acc)
    from matplotlib import pyplot as plt
    #画出正例和反例的散点图
    plt.scatter(positive_x1,positive_x2,c='red')
    plt.scatter(negetive_x1,negetive_x2,c='blue')
    #画出超平面
    line_x = np.arange(-max(positive_x1),max(positive_x1))
    line_y = line_x * (-clf.coef_[0][0] / clf.coef_[0][1]) - clf.intercept_
    plt.plot(line_x,line_y)
    plt.show()

if __name__ == '__main__':
    test_mnist()
    test()