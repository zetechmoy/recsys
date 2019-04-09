from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

import pandas as pd

ratings = pd.read_csv("./ml-100k/u.data", names=["UserId", "MovieId", "Rate", "TimeStamp"], sep="\t")
genres = pd.read_csv("./ml-100k/u.genre", names=["Genre", "GenreId"], sep="|")
movies = pd.read_csv("./ml-100k/u.item", names=["MovieId", "MovieTitle", "ReleaseDate","VideoReleaseDate", "IMDbURL", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"], sep="|")
occupations = pd.read_csv("./ml-100k/u.occupation", names=["Occupation"], sep="\t")
users = pd.read_csv("./ml-100k/u.user", names=["UserId", "Age", "Gender", "Occupation", "ZipCode"], sep="|")

movie_ratings = pd.merge(movies, ratings)
lens = pd.merge(movie_ratings, users)

columns_to_drop = ["TimeStamp", "UserId", "ReleaseDate", "VideoReleaseDate", "IMDbURL", "unknown", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
lens.drop(labels=columns_to_drop, axis=1, inplace=True)

print(lens.loc[0])
print("##########")
print(lens.loc[1])
print("##########")
print(lens.loc[2])



exit()
x = dataset["data"]
y = dataset["target"]

X_train, X_validation, y_train, y_validation = train_test_split(x, y, random_state=0)

learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=3, max_depth = 3, random_state = 0)
    gb.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_validation, y_validation)))
    print()
