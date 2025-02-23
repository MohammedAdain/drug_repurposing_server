from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from pydantic import BaseModel
import random
from starlette.responses import HTMLResponse
import pickle
from rdkit import Chem
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

from fastapi.staticfiles import StaticFiles


app = FastAPI()

# Serve static files (e.g., images, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 for rendering HTML templates
templates = Jinja2Templates(directory="templates")

# Dictionary to store drug name → SMILES mapping
drug_data = {}

# Dictionary to store SMILES → drug name mapping
drug_data_smiles_drug_name = {}

# Load CSV file and populate dictionary at startup
@app.on_event("startup")
def load_csv():
    global drug_data
    global w2v_model
    global loaded_model
    try:
        df = pd.read_csv("drugs.csv")  # Replace with your actual file path
        drug_data = dict(zip(df["Unnamed: 3"].astype(str), df["smiles"].astype(str)))
        drug_data_smiles_drug_name = dict(zip(df["smiles"].astype(str), df["Unnamed: 3"].astype(str)))
        print(f"Loaded {len(drug_data)} drugs from CSV.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
    try:
        # Load the trained RandomForestRegressor model
        model_filename = "random_forest_model.pkl"
        with open(model_filename, "rb") as file:
            loaded_model = pickle.load(file)

        print("Random Forest Model successfully loaded!")

        # Load the pre-trained Word2Vec model used for embedding
        w2v_model_path = "model_300dim.pkl"
        w2v_model = word2vec.Word2Vec.load(w2v_model_path)

        print("Word2Vec Model successfully loaded!")
    except Exception as e:
        print(f"Error loading pickel models: {e}")


# Function to convert a single SMILES string to mol2vec embedding
def smiles_to_mol2vec(smiles, model, unseen="UNK"):
    """ Converts a SMILES string to a Mol2Vec vector using Gensim 4.0+ API """
    mol = Chem.MolFromSmiles(smiles)  # Convert to RDKit Mol object
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    sentence = MolSentence(mol2alt_sentence(mol, 1))  # Generate sentence
    vector = []

    for word in sentence:
        if word in model.wv.key_to_index:
            vector.append(model.wv[word])
        elif unseen is not None:
            vector.append(model.wv[unseen] if unseen in model.wv.key_to_index else np.zeros(model.vector_size))

    return np.mean(vector, axis=0) if vector else np.zeros(model.vector_size)


# Serve the HTML form
@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# Handle form submission
@app.post("/submit/")
async def submit_form(
    request: Request,
    drug_name: str = Form(None),
    molecular_formula: str = Form(None)
):
    # Check if drug name exists in the dictionary
    smiles = drug_data.get(drug_name, "Unknown") if drug_name else "N/A"

    # if molecular_formula:
        # drug_name = drug_data_smiles_drug_name.get(molecular_formula, "Unknown") if molecular_formula else "N/A"
    
    if not drug_name:
        print("Invalid molecular_formula")
    
    # Generate a random affinity number
    # affinity = round(random.uniform(0.1, 10.0), 2)
    # Convert the SMILES input to a vector
    X_input = np.array([smiles_to_mol2vec(smiles, w2v_model)])
    print(f"Processed Input Shape: {X_input.shape}")  # Should match the shape of X during training
    # Make prediction
    y_pred = loaded_model.predict(X_input)
    print(f"Predicted vina_score: {y_pred[0]}")

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "input_type": "Drug Name" if drug_name else "Molecular Formula",
            "input_value": drug_name if drug_name else molecular_formula,
            "smiles": smiles,
            "affinity": y_pred[0]
        }
    )

# API Endpoint to get SMILES for a given drug name
@app.get("/get_smiles/{drug_name}")
async def get_smiles(drug_name: str):
    smiles = drug_data.get(drug_name)
    if smiles:
        return {"drug_name": drug_name, "smiles": smiles}
    else:
        raise HTTPException(status_code=404, detail="Drug not found")

# Run the app (only when executed directly)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)