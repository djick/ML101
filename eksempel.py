####################################################################################
# Datasett er hentet fra                                                            #
# https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/datasets/data   #
# En tutorial som går mer i dybden finnes her:                                     #
# http://machinelearningmastery.com/machine-learning-in-python-step-by-step/       #
####################################################################################
# importerer nødvendige biblioteker
from sklearn import model_selection, metrics, svm
from pandas import read_csv

## Datasettet
# Les inn data
iris = read_csv("data/iris.csv")

# Skriv ut navnet på kolonnene
print(60 * "=")
print("Kolonner:")
print(iris.columns)
print(60 * "=")

# Skriv ut matrisens størrelse
print("Form:")
print(iris.shape)
print(60 * "=")

# Skriv ut første rad
print("Første rad:")
print(iris.iloc[0])
print(60 * "=")

# skriv ut første rad i kolonnen for klassen
print("Første rads klasse:")
print(iris["class"].iloc[0])
print(60 * "=")

# Skriv ut statistisk sammendrag av data
print("Statistikk:")
print(iris.describe())
print(60 * "=")

# Skriv ut klasse distribusjon
print("Klassedistrubisjon: ")
print(iris.groupby('class').size())
print(60 * "=")

# Separer 'features' fra 'targets'
verdier = iris[["sepal length", "sepal width", "petal length", "petal width"]]
klasser = iris["class"]

# Del datasettet opp i test og trening
X_train, X_test, y_train, y_test = model_selection.train_test_split(verdier, klasser, test_size=0.33, random_state=1)

## Lag en klassifiseringsmodell
print("Modell:")
# Benytter SVM - Support Vector Machine
classifier = svm.SVC(gamma=0.001, C=100.)
print(classifier)
print(60 * "=")

# Tren modellen
classifier.fit(X_train, y_train)

# Prediker (test)
prediksjoner = classifier.predict(X_test)

## Resultater
print("Resultater:")
print(metrics.classification_report(y_test, prediksjoner, target_names=iris["class"].unique()))
print(60 * "=")

print("Confusion matrix:")
print(metrics.confusion_matrix(y_test, prediksjoner))
print(60 * "=")

## Alternativ løsning
print(60 * "#")
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=300, random_state=1)
clf.fit(X_train, y_train)

prediksjoner_MLP = clf.predict(X_test)

print(metrics.classification_report(y_test, prediksjoner_MLP, target_names=iris["class"].unique()))
print(60 * "=")
print(metrics.confusion_matrix(y_test, prediksjoner_MLP))
print(60 * "=")
