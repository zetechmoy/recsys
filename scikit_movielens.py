from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#ratings = pd.read_csv("./ml-100k/u.data", names=["UserId", "MovieId", "Rate", "TimeStamp"], sep="\t")
#genres = pd.read_csv("./ml-100k/u.genre", names=["Genre", "GenreId"], sep="|")
#movies = pd.read_csv("./ml-100k/u.item", names=["MovieId", "MovieTitle", "ReleaseDate","VideoReleaseDate", "IMDbURL", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"], sep="|")
#occupations = pd.read_csv("./ml-100k/u.occupation", names=["Occupation", "OccupationId"], sep="|")
#users = pd.read_csv("./ml-100k/u.user", names=["UserId", "Age", "Gender", "Occupation", "ZipCode"], sep="|")
#
#movie_ratings = pd.merge(movies, ratings)
#fusers = pd.merge(occupations, users)
#lens = pd.merge(movie_ratings, fusers)
#columns_to_drop = ["MovieTitle", "Occupation", "TimeStamp", "UserId", "ReleaseDate", "VideoReleaseDate", "IMDbURL", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
#lens.drop(labels=columns_to_drop, axis=1, inplace=True)
#
#lens["Gender"] = [1 if g == "M" else 0 for g in lens["Gender"]]
#print(lens)
#
#f = open("dataset.csv", "w+")
#f.write(lens.to_csv(index=False))
#f.close()

lens = pd.read_csv("dataset.csv", names=["MovieId","Rate","OccupationId","Age","Gender","ZipCode"], dtype={"MovieId":np.int64,"Rate":np.int64,"OccupationId":np.int64,"Age":np.int64,"Gender":np.int64,"ZipCode":np.int64})
seed = 7
y = []
for l in lens["Rate"]:
	liked = 1
	not_liked = 0

	lorn = liked if l >= 2.5 else not_liked

	y.append(lorn)

x = lens
x.drop(labels=["Rate"], axis=1, inplace=True)

X_train, X_validation, y_train, y_validation = train_test_split(x, y, random_state=seed)

learning_rates = [0.001]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=1000, learning_rate = learning_rate, max_depth = 3, random_state = seed)
    gb.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_validation, y_validation)))


print("#######TEST#######")
X_test = X_validation[0:10]
y_test = y_validation[0:10]
res = gb.predict_proba(X_test)
print(X_test)
print(res)
print(y_test)
