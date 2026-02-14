from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="rf_model API",
    description="API z wyuczonym modelem",
    version="1.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

with open("penguin_rf_model.pkl", "rb") as f:
    model = joblib.load(f)

print("MODEL LOADED:", type(model))

class DataInput(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/predict")
def predict(data: DataInput):
    input_df = pd.DataFrame([{
        "bill_length_mm": data.bill_length_mm,
        "bill_depth_mm": data.bill_depth_mm,
        "flipper_length_mm": data.flipper_length_mm,
    }])


    species_map = {
        0: "Adelie",
        1: "Chinstrap",
        2: "Gentoo"
    }

    image_map = {
        "Adelie": "image/adelie.jpeg",
        "Chinstrap": "image/chinstrap.jpeg",
        "Gentoo": "image/gentoo.jpeg"
    }

    link_map = {
        "Adelie": "https://ebird.org/species/adepen1",
        "Chinstrap": "https://ebird.org/species/chipen2",
        "Gentoo": "https://ebird.org/species/genpen1"
    }

    prediction_idx = int(model.predict(input_df)[0])
    species_name = species_map[prediction_idx]


    return {
        "prediction": species_name,
        "image_url": image_map[species_name],
        "info_url": link_map[species_name]
    }
