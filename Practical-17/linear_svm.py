
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score

# Load Iris (two classes, two features)
iris = load_iris()
X, y = iris.data[iris.target < 2, 2:], iris.target[iris.target < 2]

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Linear SVM
model = SVC(kernel="linear", C=1)
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Plot decision boundary and margins
DecisionBoundaryDisplay.from_estimator(
    model, X_train, response_method="decision_function",
    levels=[-1, 0, 1], linestyles=["--", "-", "--"],
    colors="black"
)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm")
plt.scatter(*model.support_vectors_.T, s=150, facecolors="none",
            edgecolors="green", label="Support Vectors")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Linear SVM - Iris Dataset")
plt.legend()
plt.show()
