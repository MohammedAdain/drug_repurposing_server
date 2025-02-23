

# from fastapi import FastAPI, Form, Request
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# import random
# from starlette.responses import HTMLResponse

# app = FastAPI()

# # Set up Jinja2 for rendering HTML templates
# templates = Jinja2Templates(directory="templates")

# # Define Drug model
# class Drug(BaseModel):
#     name: str

# # Serve the HTML form
# @app.get("/", response_class=HTMLResponse)
# async def form_page(request: Request):
#     return templates.TemplateResponse("form.html", {"request": request})

# # Handle form submission
# @app.post("/submit/")
# async def submit_form(request: Request, drug_name: str = Form(...)):
#     # Generate a random affinity number
#     affinity = round(random.uniform(0.1, 10.0), 2)
#     return templates.TemplateResponse("result.html", {"request": request, "drug_name": drug_name, "affinity": affinity})

# # Run the app (only when executed directly)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import random
from starlette.responses import HTMLResponse

app = FastAPI()

# Set up Jinja2 for rendering HTML templates
templates = Jinja2Templates(directory="templates")

# Define Drug model
class Drug(BaseModel):
    name: str = None
    molecular_formula: str = None

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
    # Generate a random affinity number
    affinity = round(random.uniform(0.1, 10.0), 2)

    # Determine which input was provided
    input_type = "Drug Name" if drug_name else "Molecular Formula"
    input_value = drug_name if drug_name else molecular_formula

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "input_type": input_type,
            "input_value": input_value,
            "affinity": affinity
        }
    )

# Run the app (only when executed directly)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)