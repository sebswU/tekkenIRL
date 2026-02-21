from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates") #jinja used to later send data from backend to frontend 

@app.get("/", response_class=HTMLResponse) #links to html file
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})