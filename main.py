import pickle
from enum import Enum

from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from category_encoders import OrdinalEncoder

app = FastAPI()


class ModelType(str, Enum):
    randomforest = "randomforest"
    logisticregression = "logisticregression"


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


data = pd.read_csv('loan_data.csv')
categorical_features = [col for col in data.columns if data[col].dtypes == 'object']
encoder = OrdinalEncoder(cols=categorical_features).fit(data)
data2 = encoder.transform(data)
data2.drop(['credit.policy', 'purpose', 'int.rate', 'days.with.cr.line'], axis=1, inplace=True)

# Traitement de certaines données
fico = data2['fico']
fico[fico >= 800] = 5
fico[fico >= 740] = 4
fico[fico >= 670] = 3
fico[fico >= 580] = 2
fico[fico >= 300] = 1
data2['fico'] = fico

revolBal = data2['revol.bal']

revolBalq1, revolBalq3 = np.percentile(revolBal, [25, 75])
revolBaliqr = revolBalq3 - revolBalq1
revolBalupper_fence = revolBalq3 + (1.5 * revolBaliqr)

revilUtil = data2['revol.util']
revilUtil[revilUtil > 100] = 100
data2['revol.util'] = revilUtil

revolBal[revolBal >= revolBalupper_fence] = revolBalupper_fence
data2['revol.bal'] = revolBal

# Scaler
#scaled_features = data.copy()
col_names = ['installment', 'log.annual.inc', 'dti', 'fico', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs',
             'pub.rec', 'not.fully.paid']
#features = scaled_features[col_names]
#scaler = StandardScaler().fit(features.values)
#features = scaler.transform(features.values)
#scaled_features[col_names] = features

X_train, X_test, y_train, y_test = train_test_split(
    data2,
    data["not.fully.paid"], test_size=0.2,
    random_state=42)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return (({"message": f"Hello {name}"}))


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


@app.get("/model/fit/{model_type}")
async def load_model_and_predict(model_type: ModelType):
    with open(f"{model_type.value}.pkl", 'rb') as file:
        loaded_model = pickle.load(file)
    loaded_model.fit(X_train, y_train)
    with open(f"{model_type.value}.pkl", 'wb') as file:
        pickle.dump(loaded_model, file)
    return {"fit model:" f"{model_type.value}"}


@app.get("/model/predict/all/{model_type}")
async def predict_all_model(model_type: ModelType):
    with open(f"{model_type.value}.pkl", 'rb') as file:
        loaded_model = pickle.load(file)
    loaded_model.predict(X_train[:100])
    return ({"Accuracy:" f"{loaded_model.score(X_train, y_train)}"})


@app.post("/model/predict/{model_type}")
async def predict_model(model_type: ModelType, input_data: InputData):
    try:
        with open(f"{model_type.value}.pkl", 'rb') as file:
            loaded_model = pickle.load(file)

            if input_data.fico >= 800:
                input_data.fico = 5
            elif input_data.fico >= 740:
                input_data.fico = 4
            elif input_data.fico >= 670:
                input_data.fico = 3
            elif input_data.fico >= 580:
                input_data.fico = 2
            else:
                input_data.fico = 1

        input_values = [
            input_data.installment, input_data.log_annual_inc, input_data.dti,
            input_data.fico, input_data.revol_bal, input_data.revol_util,
            input_data.inq_last_6mths, input_data.delinq_2yrs, input_data.pub_rec
        ]

        prediction = loaded_model.predict([input_values])[0]

        return {"prediction": bool(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")