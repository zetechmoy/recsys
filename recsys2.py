import sys, json
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

def tanimoto(a, b):
	mod_a = sum(a)
	mod_b = sum(b)
	a_and_b = sum([a[i] and b[i] for i in range(0, len(a))])
	return a_and_b/(mod_a + mod_b - a_and_b)

#print(tanimoto([1,0,1,1,0,1], [1,1,0,1,0,0]))
#[0,1,0,0,1,0]

warnings.filterwarnings('ignore')

dataset = pd.read_csv("./ml-100k/u.data", names=["user_id", "movie_id", "rating", "TimeStamp"], sep="\t")
dataset.drop(labels=["TimeStamp"], axis=1, inplace=True)
movies = pd.read_csv("./ml-100k/u.item", names=["movie_id", "MovieTitle", "ReleaseDate","VideoReleaseDate", "IMDbURL", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"], sep="|")
users = pd.read_csv("./ml-100k/u.user", names=["user_id", "Age", "Gender", "Occupation", "ZipCode"], sep="|")

if False:

	users_ratings = pd.merge(users, dataset)
	#users_ratings.drop(labels=["TimeStamp"], axis=1, inplace=True)

	genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
	user_vectors = [[0 for _ in genres] for u in range(0, len(users_ratings.user_id.unique()))]

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


tab = np.dot(user_vectors, movie_vectors.T)

user_index = 4
best_similarity_user_index = -1
best_similarity = 0

for j in range(0, user_vectors.shape[0]):
	if user_index != j:
		tan = tanimoto(user_vectors[user_index], user_vectors[j])
		if tan > best_similarity:
			best_similarity = tan
			best_similarity_user_index = j

print(user_vectors[user_index])
print(user_vectors[j])
print(tanimoto(user_vectors[user_index], user_vectors[j]))

user_rating_film = tab[j]
sim_user_rating_film = tab[user_index]

#let say user has not watched film 1 to 5, so the note the user would give will be close to the similar user
print("Predictions (similar user): ")
print(sim_user_rating_film[45:50])

print("Reality : ")
print(user_rating_film[45:50])

#To predict the best film to suggest to the user we make the prediction on every film he has not watched yet and take the best one (with higher rating)
#let's say he has watched no film => predict on everything
sim_user_rating_film_cpy = sim_user_rating_film.copy()
sim_user_rating_film_cpy.sort()
recommended_movie_rate = sim_user_rating_film_cpy[::-1][:5]
print(recommended_movie_rate)

recommended_movie_ids = sim_user_rating_film.argsort()[:5]
print(recommended_movie_ids)

print(movies[movies['movie_id'].isin(recommended_movie_ids)]['MovieTitle'])
