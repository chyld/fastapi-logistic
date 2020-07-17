# pip install fastapi uvicorn
# uvicorn main:app --reload --host 0.0.0.0 --port 3000
# site/docs
# site/redoc

import pickle

from fastapi import FastAPI

app = FastAPI()
model = pickle.load(open('model.p', 'rb'))
targets = ['setosa', 'versicolor', 'virginica']


@app.get("/")
def hello():
    return {"Hello": "World"}


@app.get("/square/{n}")
def square(n: int):
    return {"n": n, "square": n ** 2}


@app.get("/predict/{sl}/{sw}/{pl}/{pw}")
def predict(sl: float, sw: float, pl: float, pw: float):
    predictions = model.predict([[sl, sw, pl, pw]])
    species = targets[predictions[0]]
    return {'species': species}
