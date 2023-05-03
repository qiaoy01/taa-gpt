import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import json

class LossData(BaseModel):
    iteration: int
    loss: float


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

loss_data = []
max_iter = 1000
yMax = 1.0
all_loss = []

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/submit_loss_data")
async def submit_loss_data(data: LossData):
    global yMax, max_iter, loss_data, all_loss
    all_loss.append(data.loss)
    loss_data.append((data.iteration, data.loss))
    if yMax < data.loss:
        yMax = data.loss
    
    if len(loss_data) > max_iter:
        loss_data.pop(0)
    return {"status": "success"}

@app.get("/loss_data")
async def get_loss_data():
    global yMax, max_iter, loss_data, all_loss
    all_len = len(all_loss)
    d = all_len // max_iter + 1
    all_data = []
    for i in range(0, all_len, d):
        all_data.append(all_loss[i])

    return {"loss_data": [{"x": point[0], "y": point[1]} for point in loss_data[-1000:]], "all_data":all_data, "all_window": max_iter, "all_max":yMax}

    
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
