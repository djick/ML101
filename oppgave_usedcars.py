####################################################################################
# Datasett er hentet fra                                                           #
# https://github.com/stedy/Machine-Learning-with-R-datasets                        #
# følg eksempel.py og tren en model for å predikere model på en bruktbil           #
####################################################################################
# importerer nødvendige biblioteker
from sklearn import model_selection, metrics, svm
from sklearn.preprocessing import LabelEncoder
import pandas as pd

## Datasettet
# Laster dataset
usedcars = pd.read_csv("data/usedcars.csv")

# Skriv ut klasse distribusjon
print("Klassedistrubisjon: ")
print(usedcars.groupby("transmission").size())
print(60 * "=")

# Konverter kolonner med tekst til numeriske verdier
le_model = LabelEncoder()
le_model.fit(usedcars["model"])
usedcars["model"] = le_model.fit_transform(usedcars["model"])

le_color = LabelEncoder()
le_color.fit(usedcars["color"])
usedcars["color"] = le_model.fit_transform(usedcars["color"])

# Separer 'features' fra 'targets'
verdier = usedcars[["year", "model", "price", "mileage", "color"]]
klasser = usedcars["transmission"]

# Del datasettet opp i test og trening
X_train, X_test, y_train, y_test = model_selection.train_test_split(verdier, klasser, test_size=0.33, random_state=1)

## Lag en klassifiseringsmodell
print("Modell:")
# Benytter SVM - Support Vector Machine
classifier = svm.SVC(gamma=0.01, C=0.1)
print(classifier)
print(60 * "=")

# Tren modellen
classifier.fit(X_train, y_train)

# Prediker (test)
prediksjoner = classifier.predict(X_test)

## Resultater
print("Resultater:")
print(metrics.classification_report(y_test, prediksjoner, target_names=usedcars["transmission"].unique()))
print(60 * "=")

print("Confusion matrix:")
print(metrics.confusion_matrix(y_test, prediksjoner))
print(60 * "=")
