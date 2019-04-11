

#This prgram predict wether a list of film categorized by its genres has more chance to be viewed by users
import sys, os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras.models import load_model
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model

import matplotlib.pyplot as plt

#np.set_printoptions(threshold=sys.maxsize)

################################################################################
#this recommender is the final one, it's a mox between collaborative and content-based filtering cause it concatenate opinions and caracteristics of each user
#it predicts user rating in function of it's opinion about film genre he likes (with previous films he liked) and in function of it's caracteristics Age, Gender, Occupation and ZipCode
#there must be a lot of data to learn correctly and training must be very accurate

ratings = pd.read_csv("./ml-100k/u.data", names=["UserId", "MovieId", "Rate", "TimeStamp"], sep="\t")
genres = pd.read_csv("./ml-100k/u.genre", names=["Genre", "GenreId"], sep="|")
items = pd.read_csv("./ml-100k/u.item", names=["MovieId", "MovieTitle", "ReleaseDate","VideoReleaseDate", "IMDbURL", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"], sep="|")
occupations = pd.read_csv("./ml-100k/u.occupation", names=["Occupation", "OccupationId"], sep="|")
users = pd.read_csv("./ml-100k/u.user", names=["UserId", "Age", "Gender", "Occupation", "ZipCode"], sep="|")

n_users = len(ratings.UserId.unique())
n_items = len(ratings.MovieId.unique())

movie_ratings = pd.merge(items, ratings)
fusers = pd.merge(occupations, users)
lens = pd.merge(movie_ratings, fusers)
#"Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
columns_to_drop = ["MovieTitle", "OccupationId", "TimeStamp", "UserId", "ReleaseDate", "VideoReleaseDate", "IMDbURL", "unknown"]
lens.drop(labels=columns_to_drop, axis=1, inplace=True)
lens.drop(labels=["MovieId"], axis=1, inplace=True)

genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

seed = 7

print("Generating data ...")
y = lens["Rate"].values
lens.drop(labels=["Rate"], axis=1, inplace=True)
print("Ratings ... OK")

movies = lens.copy()
movies.drop(labels=["Age", "Gender", "Occupation", "ZipCode"], axis=1, inplace=True)
movies = np.array(movies)
print("Movies ... OK")

#onehot encode classical values and concatenate them each other
#/!\ since we user embedding layer in NN we don't need to onehot encode
#/!\we need to remember about the label_encode to use the model later (pickle ...)
users = None
label_encoders = []
for i in ["Gender", "Occupation", "ZipCode"]:
	label_encoder = LabelEncoder()
	feature = label_encoder.fit_transform(lens.loc[:][i])
	feature = feature.reshape(movies.shape[0], 1)
	#onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
	#feature = onehot_encoder.fit_transform(feature)
	label_encoders.append(label_encoder)
	if users is None:
		users = feature
	else:
		users = np.concatenate((users, feature), axis=1)

print("Users ... OK")
#concatenate gender, occupation, zipcode and age
ages = np.array([[i] for i in lens.loc[:]["Age"]])
users = np.hstack((users, ages))

#users are user caracteristics
#movies are user's opinion about film's genres (based on previous liked films)
#y are how much the user rated the film

#data are ready !

#movies and users are inputs, we must concatenate x and users
#movies are film liked by user genres and users are user caracteristics
#y is output
#y is how much user rate x out of 5
cut_index = movies.shape[0] - movies.shape[0]//8
movies_train, movies_test = movies[0:cut_index], movies[cut_index:movies.shape[0]]
users_train, users_test = users[0:cut_index], users[cut_index:users.shape[0]]
y_train, y_test = y[0:cut_index], y[cut_index:y.shape[0]]

if os.path.exists("recsys.h5"):
	print("A trained model has been found !")
	model = load_model("recsys.h5")
else:
	print("No trained model has been found, let's build it !")
	print("Creating a Y model with keras ...")
	# creating book embedding path
	movie_input = Input(shape=[movies.shape[1]], name="Movie-Input")
	movie_embedding = Embedding(n_users+1, 100, name="Movie-Embedding")(movie_input)
	movie_vec = Flatten(name="Flatten-Movies")(movie_embedding)

	# creating user embedding path
	user_input = Input(shape=[users.shape[1]], name="User-Input")
	user_embedding = Embedding(n_items+1, 100, name="User-Embedding")(user_input)
	user_vec = Flatten(name="Flatten-Users")(user_embedding)

	# concatenate features
	conc = Concatenate()([movie_vec, user_vec])

	# add fully-connected-layers
	fc1 = Dense(256, activation='relu')(conc)
	fc2 = Dense(128, activation='relu')(fc1)
	out = Dense(1)(fc2)

	# Create model and compile it
	model = Model([user_input, movie_input], out)
	model.compile('adam', 'mean_squared_error', metrics=['acc'])
	print("Learn ! Learn ! Learn !")
	history = model.fit([users_train, movies_train], y_train, epochs=20, verbose=1, batch_size=32)
	model.save('recsys.h5')

	plt.plot(history.history['loss'])
	plt.xlabel("Epochs")
	plt.ylabel("Training Error")
	plt.show()

#score = model.evaluate([users_test, movies_test], y_test)
#print(score)
#print('Test loss:', score[0])
#print('Test acc:', score[1])

print("################TEST################")
# Creating dataset for making recommendations for the first user (id=1)

movies_data = movies_test[0:5]#movies' genres that user has not watched yet
#print(movies_data[:50])
users_data = users_test[0:5]#[users_test[0] for _ in range(0, 50)]
#print(users_data)
#print(movies_data)

#make a prediction of how much user[i] loves books[i] and sort it
predictions = model.predict([users_data, movies_data])
predictions = np.array([a[0] for a in predictions])
print("predictions = "+str(predictions))
print("reality = "+str(y_test[0:5]))

for i in range(0, 5):
	film_genres = ','.join([genres[g] for g in range(0, len(genres)) if movies_data[i][g] == 1])
	p1 = "A "+str(label_encoders[0].inverse_transform([users_data[i][0]])[0])+" user who is "+str(label_encoders[1].inverse_transform([users_data[i][1]])[0])+" aged by "+str(users_data[i][3])+" living in "+str(label_encoders[2].inverse_transform([users_data[i][2]])[0])
	p2 = " may gives "+str(predictions[i])+" stars to a film with "+film_genres+" genres"
	print(p1+p2)
