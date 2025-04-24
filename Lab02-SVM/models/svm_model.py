from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class modelSVM:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.acc = []

    def process_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

    def train_without_scale(self):
        model1 = make_pipeline(
            LinearSVC(loss='hinge', max_iter=10000, random_state=42)
        )
        model1.fit(self.X_train, self.y_train)
        self.acc.append(model1.score(self.X_train, self.y_train))
        self.acc.append(model1.score(self.X_test, self.y_test))

    def train_with_scale(self):
        model2 = make_pipeline(
            StandardScaler(),
            LinearSVC(loss='hinge', max_iter=10000, random_state=42)
        )
        model2.fit(self.X_train, self.y_train)
        self.acc.append(model2.score(self.X_train, self.y_train))
        self.acc.append(model2.score(self.X_test, self.y_test))

    def run(self):
        self.process_data()
        self.train_without_scale()
        self.train_with_scale()
        return self.acc
