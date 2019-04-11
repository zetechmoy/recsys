import sys, json
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

def tanimoto(a, b):
	mod_a = sum(a)
	mod_b = sum(b)
	a_and_b = sum([a[i] and b[i] for i in range(0, len(a))])
	return a_and_b/(mod_a + mod_b - a_and_b)
	
################################################################################
#collaborative filtering (user to user recommendation, how much a user is similar to another user)
#it's not using ai here, just theorical application of matrix factorization and similarity (tanimoto)
#useful when there is not a lot of data cause it's just a matrix dot

#load raw dataset
dataset = pd.read_csv("../ml-100k/u.data", names=["user_id", "movie_id", "rating", "TimeStamp"], sep="\t")
dataset.drop(labels=["TimeStamp"], axis=1, inplace=True)
movies = pd.read_csv("../ml-100k/u.item", names=["movie_id", "MovieTitle", "ReleaseDate","VideoReleaseDate", "IMDbURL", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"], sep="|")
users = pd.read_csv("../ml-100k/u.user", names=["user_id", "Age", "Gender", "Occupation", "ZipCode"], sep="|")
genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

#create dataset if not exists
if not os.path.exists("dataset_recsys2.json"):

	users_ratings = pd.merge(users, dataset)
	#users_ratings.drop(labels=["TimeStamp"], axis=1, inplace=True)

	user_vectors = [[0 for _ in genres] for u in range(0, len(users_ratings.user_id.unique()))]

	#a bit long to compute...
	for i in range(0, len(users_ratings.user_id)):
		print(i)
		if users_ratings.loc[i, "rating"] >= 3:
			movie = movies.loc[movies["movie_id"] == users_ratings.loc[i, "movie_id"]]
			vector_genre = [movie["Action"].values[0], movie["Adventure"].values[0], movie["Animation"].values[0], movie["Children's"].values[0], movie["Comedy"].values[0], movie["Crime"].values[0], movie["Documentary"].values[0], movie["Drama"].values[0], movie["Fantasy"].values[0], movie["Film-Noir"].values[0], movie["Horror"].values[0], movie["Musical"].values[0], movie["Mystery"].values[0], movie["Romance"].values[0], movie["Sci-Fi"].values[0], movie["Thriller"].values[0], movie["War"].values[0], movie["Western"].values[0]]
			user_id = users_ratings.loc[i]["user_id"]-1
			for j in range(0, len(genres)):
				if user_vectors[user_id][j] == 0 and vector_genre[j] == 1:
					user_vectors[user_id][j] = 1
			#print(user_vectors[user_id])

	user_vectors = np.array(user_vectors)
	print(user_vectors.shape)

	ratings = dataset.loc[:, "rating"].values
	print(ratings.shape)

	movies.drop(labels=["movie_id", "MovieTitle", "ReleaseDate","VideoReleaseDate", "IMDbURL", "unknown"], axis=1, inplace=True)
	movie_vectors = movies.loc[:].values
	print(movie_vectors.shape)

	data = {"user_vectors" : user_vectors.tolist(), "ratings" : ratings.tolist(), "movie_vectors" : movie_vectors.tolist() }

	f = open("dataset_recsys2.json", "w")
	f.write(json.dumps(data))
	f.close()

else:
	f = open("dataset_recsys2.json", "r")
	raw = f.read()
	data = json.loads(raw)
	f.close()
	user_vectors = np.array(data["user_vectors"])
	ratings = np.array(data["ratings"])
	movie_vectors = np.array(data["movie_vectors"])

#create the matrix factorizatioon
tab = np.dot(user_vectors, movie_vectors.T)

#get the best similar user
user_index = 4
best_similarity_user_index = -1
best_similarity = 0

for j in range(0, user_vectors.shape[0]):
	if user_index != j:
		tan = tanimoto(user_vectors[user_index], user_vectors[j])
		if tan > best_similarity:
			best_similarity = tan
			best_similarity_user_index = j

print("user = "+str(user_vectors[user_index]))
print("simalar user = "+str(user_vectors[j]))
print("similarity = "+str(tanimoto(user_vectors[user_index], user_vectors[j])))

user_rating_film = tab[j]
sim_user_rating_film = tab[user_index]

#let say user has not watched film 1 to 5, so the note the user would give will be close to the similar user
print("Predictions (similar user ratings ): "+str(sim_user_rating_film[0:5]))
print("Reality rating : "+str(user_rating_film[0:5]))

#To predict the best film to suggest to the user we make the prediction on every film he has not watched yet and take the best one (with higher rating)
#let's say he has watched no film => predict on everything
sim_user_rating_film_cpy = sim_user_rating_film.copy()
sim_user_rating_film_cpy.sort()

#recommended_movie_rate is the prediction
recommended_movie_rate = sim_user_rating_film_cpy[::-1][:5]
#corresponding ids
recommended_movie_ids = sim_user_rating_film.argsort()[:5]

for i in range(0, 5):
	print("user_id "+str(user_index)+" may gives "+str(recommended_movie_rate[i])+" stars to movie_id "+str(recommended_movie_ids[i])+" => "+movies[movies['movie_id'].isin([recommended_movie_ids[i]])]['MovieTitle'].values[0])

#print(recommended_movie_rate)
#print(recommended_movie_ids)
#print(movies[movies['movie_id'].isin(recommended_movie_ids)]['MovieTitle'])
