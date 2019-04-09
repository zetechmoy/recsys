# recsys

A simple recommendation system based on gradient boosted trees

### Prerequisites

You need basic things like

```
python3
pip3
sklearn
xgboost
```

### Installing

First install python3 and pip3

```
sudo apt install python3 python-pip3
```
Then use pip3 to install sklearn and xgboost

```
pip3 install sklearn xgboost
```

## Running the tests

Simply run

```
python3 xg_movielens_oh.py
```

### Notes
* xg_movielens_oh.py => GBT with movies's genre (Tags) (With Age, Gender, Occupation a ZipCode one_hot encoded /!\ Commentaries)
* scikit_movielens.py && xg_movielens.py are the same but with different librairies (Sci-Kit Learn & XGBoost)
* To predict wether a film should be recommended or not, give the movie's genre and predict probabilities
* Give a list of film and sort following probabilities to have the best one to recommend

### References
* https://www.youtube.com/watch?v=E0vXeruvmqg
* https://www.youtube.com/watch?v=7VeUPuFGJHk
* https://www.youtube.com/watch?v=jxuNLH5dXCs

## Authors

* **Théo Guidoux** - *Everything* - [zetechmoy](https://github.com/zetechmoy)

## License

This project is licensed under the Apache2 License - see the [LICENSE.md](LICENSE.md) file for details
