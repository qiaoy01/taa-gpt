import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import json
import asyncio
import os
import subprocess
import logging


class LossData(BaseModel):
    iteration: int
    loss: float


class TrainingProcess(BaseModel):
    program: str
    config: str


app = FastAPI()
logging.basicConfig(level=logging.ERROR)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/css", StaticFiles(directory="css"), name="css")
templates = Jinja2Templates(directory="templates")

loss_data = []
max_iter = 1000
yMax = 1.0
all_loss = []
connected_clients = []


def calculate_gpt_params(json_data):
    d_model = json_data["d_model"]
    n_head = json_data["n_head"]
    d_k = json_data["d_k"]
    d_v = json_data["d_v"]
    d_ff = json_data["d_ff"]
    n_layer = json_data["n_layer"]
    vocab_size = json_data["vocab_size"]

    # Embedding matrix
    embedding_params = vocab_size * d_model

    # Multi-head self-attention layers
    attention_params = n_layer * \
        (n_head * (d_k * d_model + d_v * d_model) + d_model * d_model)

    # Position-wise feed-forward layers
    feed_forward_params = n_layer * (2 * d_model * d_ff)

    # Output layer
    output_params = d_model * vocab_size

    # Total parameters
    total_params = embedding_params + attention_params + \
        feed_forward_params + output_params

    return total_params


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


async def send_loss_data_to_clients(data):
    json_data = json.dumps({"x": data.iteration, "y": data.loss})
    for client in connected_clients:
        await client.send_text(json_data)


@app.post("/submit_loss_data")
async def submit_loss_data(data: LossData):
    global yMax, max_iter, loss_data, all_loss
    all_loss.append(data.loss)
    loss_data.append((data.iteration, data.loss))
    if yMax < data.loss:
        yMax = data.loss

    if len(loss_data) > max_iter:
        loss_data.pop(0)

    # asyncio.create_task(send_loss_data_to_clients(data))
    return {"status": "success"}


def get_matching_files(directory, start_str, end_str):
    matching_files = []

    for filename in os.listdir(directory):
        if filename.startswith(start_str) and filename.endswith(end_str):
            matching_files.append(filename)

    return matching_files


@app.get("/config_details")
async def get_config_details(name):
    with open("./configs/config_" + name + ".json", "r") as f:
        config = json.load(f)

    total_parameters = calculate_gpt_params(config)

    return {"config": config, "model_size": total_parameters, "model_size_m": total_parameters / 1048576, "total_size_g": total_parameters / 1073741824}


@app.get("/config_list")
async def get_config_list():
    config_list = get_matching_files("./configs", "config_", ".json")
    project_list = get_matching_files("./projects", "train_", ".py")
    return {"config_list": config_list, "project_list": project_list}


def run_python_script(script_path, log_file_path):
    with open(log_file_path, 'w') as log_file:
        subprocess.Popen(["python", script_path], stdout=log_file,
                         stderr=subprocess.STDOUT, start_new_session=True)


@app.post("/start_training")
async def start_training(data: TrainingProcess):
    global yMax, max_iter, loss_data, all_loss
    print("start training...")
    print(data)
    
    yMax = 1.0
    max_iter = 1000
    loss_data = []
    all_loss = []

    json_data = json.loads(data.config)

    with open("config.json", 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    run_python_script("./projects/train_" + data.program +
                      ".py", "./logs/" + data.program + ".log")


@app.get("/loss_data")
async def get_loss_data():
    global yMax, max_iter, loss_data, all_loss
    all_len = len(all_loss)
    d = all_len // max_iter + 1
    all_data = []
    for i in range(0, all_len, d):
        all_data.append(all_loss[i])

    return {"loss_data": [{"x": point[0], "y": point[1]} for point in loss_data[-1000:]], "all_data": all_data, "all_window": max_iter, "all_max": yMax}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="error", reload=True)
