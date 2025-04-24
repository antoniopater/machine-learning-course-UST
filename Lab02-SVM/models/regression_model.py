from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_squared_error

class Regression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.acc = []

    def process_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

    def linear_svr(self):
        model = make_pipeline(
            PolynomialFeatures(degree=4),
            LinearSVR(max_iter=10000)
        )
        model.fit(self.X_train, self.y_train)
        train_prediction = model.predict(self.X_train)
        test_prediction = model.predict(self.X_test)
        self.acc.append(mean_squared_error(self.y_train, train_prediction))
        self.acc.append(mean_squared_error(self.y_test, test_prediction))

    def poly_svr(self):
        model = make_pipeline(
            PolynomialFeatures(degree=4),
            SVR(kernel='poly', degree=4)
        )
        model.fit(self.X_train, self.y_train)

    def grid_svr(self):
        model = make_pipeline(
            PolynomialFeatures(degree=4),
            SVR(kernel='poly', degree=4)
        )
        param = {
            "svr__C": [0.1, 1, 10],
            "svr__coef0": [0.1, 1, 10]
        }
        grid = GridSearchCV(model, param, scoring='neg_mean_squared_error', cv=5)
        grid.fit(self.X, self.y)
        self.best_model = grid.best_estimator_
        self.best_params = grid.best_params_

    def best_params_model(self):
        self.best_model.fit(self.X_train, self.y_train)
        train_prediction = self.best_model.predict(self.X_train)
        test_prediction = self.best_model.predict(self.X_test)
        self.acc.append(mean_squared_error(self.y_train, train_prediction))
        self.acc.append(mean_squared_error(self.y_test, test_prediction))

    def run(self):
        self.process_data()
        self.linear_svr()
        self.poly_svr()
        self.grid_svr()
        self.best_params_model()
        return self.acc
