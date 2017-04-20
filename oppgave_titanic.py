####################################################################################
# Datasett er hentet fra                                                           #
# https://www.kaggle.com/c/titanic                                                 #
# følg eksempel.py og tren en model for å predikere om en person overlever.        #
# Siden testsettet som følger med ikke har fasit benytter vi oss kun av            #
# treningsdata.                                                                     #
####################################################################################
# importerer nødvendige biblioteker
from sklearn import model_selection, metrics, svm
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Laster dataset
titanic = pd.read_csv("data/titanic/train.csv")

# Henter ut klassene
labels = titanic["Survived"]
del titanic["Survived"]

# Konverter priser til hele verider
le_priser = LabelEncoder()
le_priser.fit(titanic["Fare"])
titanic["Fare"] = le_priser.fit_transform(titanic["Fare"])

# Fjern kolonner med manglende verdier
titanic = titanic.dropna(axis=1)

# Del datasettet opp i test og trening
X_train, X_test, y_train, y_test = model_selection.train_test_split(titanic._get_numeric_data(),
                                                                    labels, test_size=0.33,
                                                                    random_state=1)

## Lag en klassifiseringsmodell
# Lager en model
# Benytter SVM - Support Vector Machine
print("Modell:")
classifier = svm.SVC(gamma=0.001, C=100.)
print(classifier)
print(60 * "=")

# Tren modellen
classifier.fit(X_train, y_train)

# Prediker (test)
prediksjoner = classifier.predict(X_test)

## Resultater
print("Resultater:")
print(metrics.classification_report(y_test, prediksjoner, target_names=["Nei", "Ja"]))
print(60 * "=")

print("Confusion matrix:")
print(metrics.confusion_matrix(y_test, prediksjoner))
print(60 * "=")
