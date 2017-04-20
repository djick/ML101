####################################################################################
# Datasett og flere instruksjoner finnes her                                       #
# https://www.dataquest.io/blog/machine-learning-python/                           #
####################################################################################
# importerer nødvendige biblioteker
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pandas import read_csv

## Datasettet
# Les inn data
games = read_csv("data/games.csv")

# Skriv ut navnet på kolonnene
print(games.columns)

# Skriv ut matrisens størrelse
print(games.shape)

# Lag et histogram for average_ratings
plt.hist(games["average_rating"])

# Vis graf
plt.show()

# Fjerner rader uten reviews.
games = games[games["users_rated"] > 0]
# Fjerner rader med manglende verdier
games = games.dropna(axis=0)

# Print matrisens størrelse
print(games.shape)

## Klustering
# KMeans model
kmeans_model = KMeans(n_clusters=5, random_state=1)
# Hent numeriske kolonner
good_columns = games._get_numeric_data()
# Tren
kmeans_model.fit(good_columns)
# Hent resultatet av clustering
labels = kmeans_model.labels_

# Lager modell for å redusere dimensjoner
pca_2 = PCA(2)
# Utfør dimmensjonsreduksjon
plot_columns = pca_2.fit_transform(good_columns)
# Scatter plot med clustrene
plt.scatter(x=plot_columns[:, 0], y=plot_columns[:, 1], c=labels)
# Vis plot
plt.show()
