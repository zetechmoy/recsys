from sklearn.datasets import load_iris

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import numpy as np

iris=load_iris()

x = iris["data"]
y = iris["target"]

X_train, X_validation, y_train, y_validation = train_test_split(x, y, random_state=0)
print(np.array(X_train).shape)
print(np.array(y_train).shape)

learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=3, max_depth = 3, random_state = 0)
    gb.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_validation, y_validation)))
    print()
