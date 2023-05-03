import requests
import time
import random

def submit_loss_data(iteration, loss):
    url = "http://127.0.0.1:8000/submit_loss_data"
    data = {"iteration": iteration, "loss": loss}
    response = requests.post(url, json=data)
    return response.json()

if __name__ == "__main__":
    for i in range(10000):
        loss = random.uniform(0, 1)
        response = submit_loss_data(i, loss)
        print(f"Submitted data for iteration {i}, status: {response['status']}")
        time.sleep(1)
