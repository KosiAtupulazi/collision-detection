from fastapi import FastAPI

app = FastAPI()

@app.get("/")

def root():
    return {"Message" : "Collision Detection page is working"}

@app.get("/demo")

def detect_demo():
        return {"Message" : "Demo page is working"}
