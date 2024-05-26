import keras
# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import cross_origin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
ann = keras.models.Sequential()
sc = StandardScaler()
np.set_printoptions(threshold=30)
geography_array = []
genders = []


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():  # put application's code here
    data = request.json

    country = str(data.get('country')).split(",")
    prediction_result = ann.predict(sc.transform(
        [[float(country[0]), float(country[1]), float(country[2]), data.get('credit_score'), int(data.get('gender')),
          data.get('age'),
          data.get('tenure'), float(data.get('balance')), data.get('number_of_products'), int(data.get('credit_card')),
          int(data.get('active_member')),
          float(data.get('estimated_salary'))]])) > 0.5
    return {"data": str(prediction_result[0][0])}


@app.route('/get-geography')
@cross_origin()
def getGeography():
    return {'data': geography_array}


@app.route('/get-gender')
@cross_origin()
def getGender():
    return {'data': genders}


if __name__ == '__main__':
    dataset = pd.read_csv('Churn_Modelling.csv')
    X = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1].values

    # Encoding categorical data
    # Label Encoding the "Gender" column
    gender_old = X[:, 2].copy()
    print(gender_old)
    le = LabelEncoder()
    X[:, 2] = le.fit_transform(X[:, 2])

    for i in range(len(gender_old)):
        if len(genders) < 2:
            if not ({"nama_gender": gender_old[i], "code": X[:, 2][i]} in genders):
                genders.append({"nama_gender": gender_old[i], "code": X[:, 2][i]})

        else:
            break

    # One Hot Encoding the "Geography" column
    x_old = X
    print(x_old[0][1])
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    print(X[0][0], X[0][1], X[0][2])

    for i in range(len(x_old)):
        if len(geography_array) < 3:
            if not ({"nama_country": str(x_old[i][1]), "kode_country": [X[i][0], X[i][1], X[i][2]]} in geography_array):
                geography_array.append({"nama_country": str(x_old[i][1]), "kode_country": [X[i][0], X[i][1], X[i][2]]})
        else:
            break

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature Scaling
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Part 2 - Building the ANN

    # Initializing the ANN
    ann = keras.models.Sequential()

    # Adding the input layer and the first hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    # Adding the second hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    # Adding the output layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Part 3 - Training the ANN

    # Compiling the ANN
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training the ANN on the Training set
    ann.fit(X_train, y_train, batch_size=32, epochs=100)
    app.run()
