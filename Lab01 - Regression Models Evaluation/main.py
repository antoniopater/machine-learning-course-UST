import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from models.regresor import Regressions
from models.save_models import save_mse, save_models

def generate_data(size=300, file_path="data/dane_do_regresji.csv"):
    X = np.random.rand(size) * 5 - 2.5
    w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
    y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8 - 4
    df = pd.DataFrame({'x': X, 'y': y})
    df.to_csv(file_path, index=False)
    df.plot.scatter(x='x', y='y')
    plt.title("Dane do regresji")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    generate_data()
    models = Regressions()
    models.load_data()
    models.linear_model()
    models.knn_model(3)
    models.knn_model(5)
    for degree in range(2, 6):
        models.poly_model(degree)
    df_mse = save_mse(models)
    print(df_mse)
    save_models(models.models_list)
