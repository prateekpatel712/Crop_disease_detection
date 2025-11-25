 
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message":"Crop Disease Detector API is running"}
    
    