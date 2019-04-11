import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

from sklearn.model_selection import train_test_split

from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model
from keras.models import load_model

################################################################################
#This prgm learn how much a user (caracterised by it's id) will rate a movie (carecterised by it's id too)
#x are user id
#y are movie id
#this is collaborative filtering because it compares users opinions genres about films to give the best film to recommend
# collaborative filtering => if a person A has the same opinion as a person B on an issue, A is more likely to have B's opinion on a different issue than that of a randomly chosen person

dataset = pd.read_csv("../ml-100k/u.data", names=["user_id", "movie_id", "rating", "TimeStamp"], sep="\t")
movies = pd.read_csv("../ml-100k/u.item", names=["movie_id", "MovieTitle", "ReleaseDate","VideoReleaseDate", "IMDbURL", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"], sep="|")

dataset.drop(labels=["TimeStamp"], axis=1, inplace=True)

warnings.filterwarnings('ignore')

train, test = train_test_split(dataset, test_size=0.2, random_state=42)
n_users = len(dataset.user_id.unique())
n_books = len(dataset.movie_id.unique())

# creating book embedding path
book_input = Input(shape=[1], name="Book-Input")
book_embedding = Embedding(n_books+1, 10, name="Book-Embedding")(book_input)
book_vec = Flatten(name="Flatten-Books")(book_embedding)

# creating user embedding path
user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_users+1, 10, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)

# concatenate features
conc = Concatenate()([book_vec, user_vec])

# add fully-connected-layers
fc1 = Dense(128, activation='relu')(conc)
fc2 = Dense(32, activation='relu')(fc1)
out = Dense(1)(fc2)

# Create model and compile it
model2 = Model([user_input, book_input], out)
model2.compile('adam', 'mean_squared_error')

model_file_name = 'regression_model2_e10.h5'

if os.path.exists(model_file_name):
	print("A trained model has been found !")
	model2 = load_model(model_file_name)
else:
	print("No trained model has been found, let's build it !")
	history = model2.fit([train.user_id, train.movie_id], train.rating, epochs=5, verbose=1)
	model2.save(model_file_name)
	plt.plot(history.history['loss'])
	plt.xlabel("Epochs")
	plt.ylabel("Training Error")
	#plt.show()

print("################TEST################")
# Creating dataset for making recommendations for the first user (id=1)
user_id = 4
movie_data = np.array(list(set(dataset.movie_id)))
user = np.array([user_id for i in range(len(movie_data))])

#make a prediction of how much user[i] loves books[i] and sort it
predictions = model2.predict([user, movie_data])
predictions = np.array([a[0] for a in predictions])

#sort 5 first highest predicted ratings
recommended_movie_ids = (-predictions).argsort()[:5]

for i in range(0, 5):
	#print(recommended_movie_ids)
	#get id of recommended movie
	#print(predictions[recommended_movie_ids])
	print("user_id "+str(user_id)+" may gives "+str(predictions[recommended_movie_ids[i]])+" stars to movie_id "+str(recommended_movie_ids[i])+" => "+movies[movies['movie_id'].isin([recommended_movie_ids[i]])]['MovieTitle'].values[0])

#print("Recommended movies for user_id = "+str(user_id)+" are : ")
#print(movies[movies['movie_id'].isin(recommended_movie_ids)]['MovieTitle'])
