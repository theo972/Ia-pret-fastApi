import pickle
from enum import Enum

from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# Liste des modèles
class ModelType(str, Enum):
    randomforest = "randomforest"
    logisticregression = "logisticregression"

" format des colonnes (eq formdata)"
class InputData(BaseModel):
    installment: float
    log_annual_inc: float
    dti: float
    fico: int
    revol_bal: int
    revol_util: float
    inq_last_6mths: int
    delinq_2yrs: int
    pub_rec: int

# Entrainement de la data, retirer les colonnes et demande à se baser sur not.fully.paid
data = pd.read_csv('loan_data.csv')
X_train, X_test, y_train, y_test = train_test_split(
    data[["installment", "log.annual.inc", "dti", "fico", "revol.bal", "revol.util", "inq.last.6mths", "delinq.2yrs", "pub.rec"]],
    data["not.fully.paid"], test_size=0.2,
    random_state=42)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return (({"message": f"Hello {name}"}))


# creation
@app.get("/model/create/{model_type}")
async def create_model(model_type: ModelType):
    if model_type == ModelType.randomforest:
        model = RandomForestClassifier(n_estimators=25, random_state=42, max_features=3)

    if model_type == ModelType.logisticregression:
        model = LogisticRegression(random_state=42)

    model_filename = f"{model_type.value}.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    return {"created_model": model_type.value}

# entrainement
@app.get("/model/fit/{model_type}")
async def load_model_and_predict(model_type: ModelType):
    with open(f"{model_type.value}.pkl", 'rb') as file:
        loaded_model = pickle.load(file)
    loaded_model.fit(X_train, y_train)
    with open(f"{model_type.value}.pkl", 'wb') as file:
        pickle.dump(loaded_model, file)
    return {"fit model:" f"{model_type.value}"}

# Predict d'entrainement
@app.get("/model/predict/all/{model_type}")
async def predict_all_model(model_type: ModelType):
    with open(f"{model_type.value}.pkl", 'rb') as file:
        loaded_model = pickle.load(file) # load model
    loaded_model.predict(X_train[:100])
    return ({"Accuracy:" f"{loaded_model.score(X_train, y_train)}"}) # save model

# predict du user
@app.post("/model/predict/{model_type}")
async def predict_model(model_type: ModelType, input_data: InputData):
    try:
        with open(f"{model_type.value}.pkl", 'rb') as file:
            loaded_model = pickle.load(file)

        input_values = [
            input_data.installment, input_data.log_annual_inc, input_data.dti,
            input_data.fico, input_data.revol_bal, input_data.revol_util,
            input_data.inq_last_6mths, input_data.delinq_2yrs, input_data.pub_rec
        ]
        prediction = loaded_model.predict([input_values])[0]

        return {"prediction": bool(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

