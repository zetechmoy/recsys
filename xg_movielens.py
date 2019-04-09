# plot decision tree
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random

# load data
lens = pd.read_csv("dataset.csv", names=["MovieId","Rate","OccupationId","Age","Gender","ZipCode"], dtype={"MovieId":np.int64,"Rate":np.int64,"OccupationId":np.int64,"Age":np.int64,"Gender":np.int64,"ZipCode":np.int64})
lens.drop(labels=["MovieId"], axis=1, inplace=True)

y = []
for l in lens["Rate"]:
	liked = 1
	not_liked = 0

	lorn = liked if l >= 2.5 else not_liked

	y.append(lorn)

x = lens
x.drop(labels=["Rate"], axis=1, inplace=True)

X_train, X_validation, y_train, y_validation = train_test_split(x, y, shuffle=True)

# fit model no training data
model = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=200)
model.fit(x, y)

print("Accuracy score (training): {0:.3f}".format(model.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(model.score(X_validation, y_validation)))

print("#######TEST#######")
X_test = X_validation[0:10]
y_test = y_validation[0:10]
res = model.predict_proba(X_test)
print(X_test)
print(res)
print(y_test)

# plot single tree
plot_tree(model)
plt.show()
