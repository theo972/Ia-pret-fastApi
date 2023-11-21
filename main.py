import pickle
from typing import Dict

from fastapi import FastAPI
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return (({"message": f"Hello {name}"}))

@app.get("/model/create/{name}")
async def create_model(name: str):
    data = pd.read_csv('loan_data.csv')
    print(data.head())
    data = pd.get_dummies(data, columns=['purpose'], drop_first=True)

    X = data.drop('not.fully.paid', axis=1)
    y = data['not.fully.paid']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))

    with open(f"{name}.pkl", 'wb') as file:
        pickle.dump(model, file)

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d", cmap="Blues")
    # plt.title("Confusion Matrix")
    # plt.show()
    return ({"Accuracy:" f"{accuracy_score(y_test, predictions)}"})

@app.get("/model/load/{name}")
async def load_model_and_predict(name: str):
    with open(f"{name}.pkl", 'rb') as file:
        loaded_model = pickle.load(file)
    data = pd.read_csv('loan_data.csv')
    data = pd.get_dummies(data, columns=['purpose'], drop_first=True)

    X = data.drop('not.fully.paid', axis=1)
    y = data['not.fully.paid']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predictions = loaded_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return {"AccuracyModel": accuracy}