
# Practical 17: Linear SVM using Iris Dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

# Select two classes and two features
X = iris.data[iris.target != 2, 2:4]
y = iris.target[iris.target != 2]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Linear SVM
model = SVC(kernel="linear", C=1.0)
model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred))

# Create a mesh grid for decision boundary
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualize decision boundary, margins and support vectors
plt.contour(
    xx, yy, Z,
    levels=[-1, 0, 1],
    linestyles=["--", "-", "--"],
    colors="black"
)

plt.scatter(
    X_train[:, 0], X_train[:, 1],
    c=y_train, cmap="coolwarm", edgecolors="k"
)

# Highlight support vectors
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=150, facecolors="none", edgecolors="green",
    label="Support Vectors"
)

plt.xlabel("Petal Length (Standardized)")
plt.ylabel("Petal Width (Standardized)")
plt.title("Linear SVM - Iris Dataset")
plt.legend()
plt.show()
