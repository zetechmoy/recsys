# recsys

These are implementations of collaborative filtering, content-based filtering and a mix of both (recsys.py).
More explanations below and in programs

## Collaborative Filtering

### colfil.py & colfil2.py
These recommenders learns how much a user (caracterised by it's id) will rate a movie (caracterised by it's id too).

* colfil.py : It's an ai with a Y model (user_id concatenated with movie_id), user are id and movies are id too.
* colfil2.py : It's not using ai here, just theorical application of matrix factorization and binary similarity (tanimoto). Tt's useful when there is not a lot of data because it's just a matrix dot but it's really less accurate.

```
>>> python3 colfil.py

Output :
user_id 4 may gives 5.1138506 stars to movie_id 1188 => Young Guns II (1990)
```

```
>>> python3 colfil2.py

Output :
user_id 4 may gives 5 stars to movie_id 1372 => Stranger, The (1994)
```

## Content-Based Filtering

### scikit_movielens.py

This recommender learns how many chances a user (caracterised by it's age, gender, location and occupation) can like a film. This is useful when there is not a lot of data and give quite good results.

```
>>> python3 scikit_movielens.py

Output :
A Male user who is programmer aged by 30 living in 6355 has 81.0% of chance to like "Twelve Monkeys (1995)"
```

## Collaborative and Content-Based Filtering

### recsys.py
This recommender is the final one, it's a mix between collaborative and content-based filtering because it concatenates opinions and caracteristics of each users. This is the most useful recommender.
It predicts user rating in function of it's opinion about film genre he likes (with previous films he liked) and in function of it's caracteristics (Age, Gender, Occupation and ZipCode).
There must be a lot of data to learn correctly and training must be very accurate.

```
>>> python3 recsys.py

Output :
A F user who is educator aged by 50 living in 98027 may gives 3.6091807 stars to a film with Comedy,Romance genres => Get Shorty (1995)
```

### Prerequisites

You need basic things like

```
python3
pip3
sklearn
tensorflow
keras
pandas
```

### Installing

First install python3 and pip3

```
sudo apt install python3 python-pip3
```
Then use pip3 to install sklearn and xgboost

```
pip3 install pandas sklearn tensorflow keras
```

### References
* https://www.youtube.com/watch?v=E0vXeruvmqg
* https://www.youtube.com/watch?v=7VeUPuFGJHk
* https://www.youtube.com/watch?v=jxuNLH5dXCs
* https://www.youtube.com/watch?v=BDJmJnrlaO8
* https://www.youtube.com/watch?v=ZspR5PZemcs
* https://github.com/yanneta/pytorch-tutorials/blob/master/collaborative-filtering-nn.ipynb
* https://github.com/chen0040/keras-recommender
* https://github.com/TannerGilbert/Tutorials/tree/master/Recommendation%20System


## Authors

* **Théo Guidoux** - *Everything* - [zetechmoy](https://github.com/zetechmoy) - [@TGuidoux](https://twitter.com/TGuidoux)

## License

This project is licensed under the Apache2 License - see the [LICENSE.md](LICENSE.md) file for details
