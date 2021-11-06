from __future__ import annotations
import subprocess
from dataclasses import dataclass
from functools import wraps
from typing import Any
from argparse import ArgumentParser
import datetime
import json
import logging
import os
import pickle
import pprint
import shutil
import time
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from pelutils import log, DataStorage
from pydantic import BaseModel

import train
import model as symbolic_model
import matplotlib.pyplot as plt
import numpy as np

import state as s


p = ArgumentParser()
p.add_argument("--port", type=int, default=6971)
p.add_argument("--no-train", action="store_true")
p.add_argument("--selenium", action="store_true")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

PORT = a.port
TRAIN = not a.no_train
# imdir = "grids"
# shutil.rmtree(imdir)
# os.makedirs(imdir)

start_time = time.time()
app = Flask(__name__)
Api(app)
CORS(app)

if TRAIN:
    model = train.DeepQ(train=True)
else:
    with open("model-%s.pkl" % PORT) as f:
        model = pickle.load(f)
episode_actions = list()
episode_states = list()

if a.selenium:
    cmd = " ".join(("python3", "web_driver.py", str(PORT), str(a.seed)))
    proc = subprocess.Popen([cmd], shell=True,
             stdin=None, stdout=None, stderr=None, close_fds=True)

@dataclass
class Data(DataStorage):
    rewards: list[float]
    loss: list[float]
    dist: list[int]
    timesteps: list[int]
train_data = Data([], [], [], [])

class PredictResponse(BaseModel):
    action: s.ActionType

def get_uptime() -> str:
    return '{}'.format(datetime.timedelta(seconds=time.time() - start_time))

def _get_data() -> dict[str, Any]:
    """ Returns data from a post request. Assumes json """
    # Return a dict parsed from json if possible
    if request.form:
        return request.form.to_dict()
    # Else parse raw data directly
    return json.loads(request.data.decode("utf-8"))

def api_fun(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with log.log_errors:
            # log("Received call to %s" % func.__name__)
            res = func(*args, **kwargs)
            if isinstance(res, PredictResponse):
                return jsonify(json.loads(res.json()))
            else:
                return jsonify(res)
    return wrapper

@app.route("/api")
@api_fun
def api():
    return {
        "uptime": get_uptime(),
        "service": "racing-cars",
    }

@app.route("/api/predict_", methods=["POST"])
@api_fun
def predict():
    # actions = [train.ActionType.ACCELERATE, train.ActionType.DECELERATE,
    #     train.ActionType.STEER_LEFT, train.ActionType.STEER_RIGHT,
    #     train.ActionType.NOTHING]
    # return PredictResponse(action=train.ActionType.ACCELERATE)
    pass

def _plot_grid(n, action: s.ActionType, grid: np.ndarray, velocities: np.ndarray):
    grid = grid.copy()
    grid *= 255
    # breakpoint()
    grid[(grid==0).all(axis=2)] = 255
    plt.figure(figsize=(20/2, 8.75/2))
    plt.imshow(grid)
    plt.title(action + "\n" + str(velocities.tolist()))
    savedir = os.path.join(imdir, "episode-%i" % episode_number)
    os.makedirs(savedir, exist_ok=True)
    plt.savefig(os.path.join(savedir, "state-%i" % n))
    plt.close()



# grid, velocities = state.grid_representation()
    # if len(episode_actions) % 10 == 0 or info.did_crash:
    #     _plot_grid(len(episode_actions)-1, action, grid, velocities)
    #     dat.save("episode-%i" % episode_number)
dat: s.Data
episode_number = -1
state: s.State
@app.route("/api/predict", methods=["POST"])
@api_fun
def predict_train():
    global episode_actions, episode_states, model, num_episodes, prev_info, state, episode_number, dat
    data = _get_data()
    if episode_number == -1:
        episode_number = 0
        return PredictResponse(action=s.ActionType.NOTHING)

    info = s.Information.from_dict(data)

    if not episode_actions:
        dat = s.Data([],[],[],[],[],[],[],[],[],[],[])
        state = s.State(0, s.Vector(0., 0.), 425, list(), info)
        action = s.ActionType.ACCELERATE
    else:
        state = state.new_state(info)
        action = symbolic_model.act(state)
        print(action)

    episode_actions.append(action)
    episode_states.append(state)

    if info.did_crash:
        episode_states = list()
        episode_actions = list()
        episode_number += 1
        log(f"Distance acquired: {state.info.distance}")

    return PredictResponse(action=action)

if __name__ == "__main__":
    log.configure(
        f"racing-cars{PORT}.log",
        "Racing cars API",
        log_commit=True,
        append=False,
    )
    app.run(host="0.0.0.0", port=PORT, debug=not "Tue")
