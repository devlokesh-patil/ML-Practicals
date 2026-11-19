from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=200, n_features=2,
                           n_redundant=0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

for d in [1, 2, 3, 5, None]:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)

    print("Depth:", d)
    print("Train Accuracy:", accuracy_score(y_train, model.predict(X_train)))
    print("Test Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Pruning
model = DecisionTreeClassifier(ccp_alpha=0.02, random_state=42)
model.fit(X_train, y_train)

print("Pruned Train Accuracy:", accuracy_score(y_train, model.predict(X_train)))
print("Pruned Test Accuracy:", accuracy_score(y_test, model.predict(X_test)))