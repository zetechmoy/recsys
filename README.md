# recsys
A simple recommendation system based on gradient boosted trees

References :
https://www.youtube.com/watch?v=E0vXeruvmqg
https://www.youtube.com/watch?v=7VeUPuFGJHk
https://www.youtube.com/watch?v=jxuNLH5dXCs

Working programs :
xg_movielens_oh.py => GBT with movies's genre (Tags) (With Age, Gender, Occupation a ZipCode one_hot encoded /!\ Commentaries)
scikit_movielens.py && xg_movielens.py are the same but with different librairies (Sci-Kit Learn & XGBoost)

To predict wether a film should be recommended or not, give the movie's genre and predict probabilities
Give a list of film and sort following probabilities to have the best one to recommend
