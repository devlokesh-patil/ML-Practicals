
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

# Select two classes
X = X[y != 2]
y = y[y != 2]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Linear SVM
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot decision boundary and margins
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
    np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100)
)

Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.contour(xx, yy, Z, levels=[-1, 0, 1],
            linestyles=['--', '-', '--'])

plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='red')

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear SVM")
plt.show()
