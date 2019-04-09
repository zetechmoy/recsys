

#This prgram predict wether a list of film categorized by its genres has more chance to be viewed by users

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

ratings = pd.read_csv("./ml-100k/u.data", names=["UserId", "MovieId", "Rate", "TimeStamp"], sep="\t")
genres = pd.read_csv("./ml-100k/u.genre", names=["Genre", "GenreId"], sep="|")
movies = pd.read_csv("./ml-100k/u.item", names=["MovieId", "MovieTitle", "ReleaseDate","VideoReleaseDate", "IMDbURL", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"], sep="|")
occupations = pd.read_csv("./ml-100k/u.occupation", names=["Occupation", "OccupationId"], sep="|")
users = pd.read_csv("./ml-100k/u.user", names=["UserId", "Age", "Gender", "Occupation", "ZipCode"], sep="|")

movie_ratings = pd.merge(movies, ratings)
fusers = pd.merge(occupations, users)
lens = pd.merge(movie_ratings, fusers)
#"Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
columns_to_drop = ["MovieTitle", "OccupationId", "TimeStamp", "UserId", "ReleaseDate", "VideoReleaseDate", "IMDbURL", "unknown"]
lens.drop(labels=columns_to_drop, axis=1, inplace=True)

lens.drop(labels=["MovieId"], axis=1, inplace=True)
print(lens)
seed = 7

y = []
for l in lens["Rate"]:
	liked = 1
	not_liked = 0

	lorn = liked if l >= 3 else not_liked

	y.append(lorn)

x = lens
x.drop(labels=["Rate"], axis=1, inplace=True)
x = np.array(x)

#onehot encode classical values and concatenate them each other
encoded_x = None

for i in [18, 20, 21]:
	label_encoder = LabelEncoder()
	feature = label_encoder.fit_transform(x[:,i])
	feature = feature.reshape(x.shape[0], 1)
	onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
	feature = onehot_encoder.fit_transform(feature)
	if encoded_x is None:
		encoded_x = feature
	else:
		encoded_x = np.concatenate((encoded_x, feature), axis=1)
#commment it out to add info about user who rated in input (Age, Gender, Occupation and ZipCode)
#without those features, the algo only classify in function of movies' genres (tag for tiptappro)
#encoded_x = np.concatenate((encoded_x, x[:,0:18]), axis=1)
#encoded_x = np.concatenate((encoded_x, x[:,19:20]), axis=1)

#data are ready !

X_train, X_validation, y_train, y_validation = train_test_split(encoded_x, y, random_state=seed)

learning_rates = [0.1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=10, learning_rate = learning_rate, max_depth = 3, random_state = seed)
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
