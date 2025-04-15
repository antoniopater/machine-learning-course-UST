import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

class Regressions:
    def __init__(self, test_size=0.2, data_path="data/dane_do_regresji.csv"):
        self.df = pd.read_csv(data_path)
        self.X = self.df[['x']]
        self.y = self.df['y']
        self.test_size = test_size
        self.models = {}
        self.mse_models = {}
        self.models_list = []

    def load_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42
        )
        print(f"Rozmiar danych treningowych: {self.X_train.shape[0]}")
        print(f"Rozmiar danych testowych: {self.X_test.shape[0]}")

    def linear_model(self, name="lin_reg"):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        self._evaluate_model(model, name)

    def knn_model(self, n_neighbors=3, name=None):
        name = name or f"knn_{n_neighbors}_reg"
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        model.fit(self.X_train, self.y_train)
        self._evaluate_model(model, name)

    def poly_model(self, degree=3, name=None):
        name = name or f"poly_{degree}_reg"
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(self.X_train)
        X_test_poly = poly.transform(self.X_test)

        model = LinearRegression()
        model.fit(X_train_poly, self.y_train)

        self.models[name] = model
        self.mse_models[name] = {
            "train_mse": mean_squared_error(self.y_train, model.predict(X_train_poly)),
            "test_mse": mean_squared_error(self.y_test, model.predict(X_test_poly))
        }
        self.models_list.append((model, poly))

    def _evaluate_model(self, model, name):
        model.fit(self.X_train, self.y_train)
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        self.models[name] = model
        self.mse_models[name] = {
            'train_mse': mean_squared_error(self.y_train, y_train_pred),
            'test_mse': mean_squared_error(self.y_test, y_test_pred)
        }
        self.models_list.append((model, None))
