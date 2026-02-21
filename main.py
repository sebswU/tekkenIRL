from fastapi import FastAPI
app = FastAPI()

@app.get("/")

def read_root():
    return {"status": "ok"}

#fast api will later store frames sent from the index html via jinja2 
#will run mediapipe and open cv later  
