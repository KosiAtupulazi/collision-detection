from fastapi import FastAPI
import os
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/")

def root():
    return {"Message" : "Collision Detection page is working"}

@app.get("/demo")

def detect_demo():
        return {"Message" : "Demo page is working"}
