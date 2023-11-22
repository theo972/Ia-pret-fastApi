import pickle
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from category_encoders import OrdinalEncoder

app = FastAPI()


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
data.drop(['credit.policy', 'purpose', 'int.rate', 'days.with.cr.line'], axis=1, inplace=True)
categorical_features = [col for col in data.columns if data[col].dtypes == 'object']
encoder = OrdinalEncoder(cols=categorical_features).fit(data)
data2 = encoder.transform(data)
data2.drop("not.fully.paid", axis=1, inplace=True)

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


@app.get("/model/create/{name}")
async def create_model(name: str):
    model = RandomForestClassifier(n_estimators=25, random_state=42, max_features=3)
    with open(f"{name}.pkl", 'wb') as file:
        pickle.dump(model, file)
    return ({"create model:" f"{name}"})


@app.get("/model/fit/{name}")
async def load_model_and_predict(name: str):
    with open(f"{name}.pkl", 'rb') as file:
        loaded_model = pickle.load(file)
    loaded_model.fit(X_train, y_train)
    with open(f"{name}.pkl", 'wb') as file:
        pickle.dump(loaded_model, file)
    return {"fit model:" f"{name}"}


@app.get("/model/predict/all/{name}")
async def predict_all_model(name: str):
    with open(f"{name}.pkl", 'rb') as file:
        loaded_model = pickle.load(file)
    loaded_model.predict(X_train[:100])
    return ({"Accuracy:" f"{loaded_model.score(X_train, y_train)}"})


@app.post("/model/predict/{name}")
async def predict_model(name: str, input_data: InputData):
    try:
        with open(f"{name}.pkl", 'rb') as file:
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
