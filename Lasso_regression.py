import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

class LassoRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, alpha=1.0):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            self._gradient_descent(X, y, n_samples)

    def _gradient_descent(self, X, y, n_samples):
        y_predicted = np.dot(X, self.weights) + self.bias

        # Compute gradients
        dw = (1/n_samples) * (np.dot(X.T, (y_predicted - y)) + self.alpha * np.sign(self.weights))
        db = (1/n_samples) * np.sum(y_predicted - y)

        # Update parameters
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def predict(self, X):
        return X.dot(self.weights) + self.bias


def main():
    # Importing dataset
    df = pd.read_csv("salary_data.csv")
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, 1].values

    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

    # Model training
    model = LassoRegression(iterations=1000, learning_rate=0.01, l2_penality=1)
    model.fit(X_train, Y_train)

    # Prediction on test set
    Y_pred = model.predict(X_test)
    print("Predicted values ", np.round(Y_pred[:3], 2))
    print("Real values	 ", Y_test[:3])
    print("Trained W	 ", round(model.W[0], 2))
    print("Trained b	 ", round(model.b, 2))

    # Visualization on test set
    plt.scatter(X_test, Y_test, color='blue')
    plt.plot(X_test, Y_pred, color='orange')
    plt.title('Salary vs Experience')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()


if __name__ == "__main__":
    main()
