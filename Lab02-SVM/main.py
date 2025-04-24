import pickle
import numpy as np
import pandas as pd
from sklearn import datasets
from models.svm_model import modelSVM
from models.regression_model import Regression

if __name__ == "__main__":
    # Breast Cancer Dataset
    data = datasets.load_breast_cancer(as_frame=True)
    X = data.data[['mean area', 'mean smoothness']].values
    y = data.target.values
    bc = modelSVM(X, y)
    acc = bc.run()
    print(acc)
    with open("bc_acc.pkl", "wb") as f:
        pickle.dump(acc, f)

    # Iris Dataset
    data_iris = datasets.load_iris(as_frame=True)
    X_iris = data_iris.data[['petal length (cm)', 'petal width (cm)']].values
    y_iris = (data_iris.target == 2)
    iris = modelSVM(X_iris, y_iris)
    acc_iris = iris.run()
    print(acc_iris)
    with open("iris_acc.pkl", "wb") as f:
        pickle.dump(acc_iris, f)

    # Synthetic Regression Data
    size = 900
    X_rand = np.random.rand(size) * 5 - 2.5
    w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
    y_rand = w4 * (X_rand ** 4) + w3 * (X_rand ** 3) + w2 * (X_rand ** 2) + w1 * X_rand + w0 + np.random.randn(size) * 8 - 4
    df = pd.DataFrame({'x': X_rand, 'y': y_rand})
    df.plot.scatter(x='x', y='y')
    X_rand = X_rand.reshape(-1, 1)
    model_random = Regression(X_rand, y_rand)
    acc_random = model_random.run()
    with open("reg_mse.pkl", "wb") as f:
        pickle.dump(acc_random, f)
    print(acc_random)
