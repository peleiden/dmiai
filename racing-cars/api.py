from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, List
from argparse import ArgumentParser
import datetime
import json
import logging
import pickle
import time
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from pelutils import log, DataStorage
from pydantic import BaseModel
import numpy as np


import train

p = ArgumentParser()
p.add_argument("--port", type=int, default=6971)

PORT = p.parse_args().port
TRAIN = True

start_time = time.time()
app = Flask(__name__)
Api(app)
CORS(app)

if TRAIN:
    model = train.DeepQ(train=True)
else:
    with open("model-%s.pkl" % PORT) as f:
        model = pickle.load(f)
num_episodes = 0
episode_actions = list()
episode_states = list()


@dataclass
class Data(DataStorage):
    rewards: list[float]
    loss: list[float]
    dist: list[int]
    timesteps: list[int]
train_data = Data([], [], [], [])

class PredictResponse(BaseModel):
    action: train.ActionType

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

@app.route("/api/predict", methods=["POST"])
@api_fun
def predict_train():
    global episode_actions, episode_states, model, num_episodes
    data = _get_data()

    state = train.State.from_dict(data)
    action = model.predict(state)
    # log("Making action %s" % action)
    episode_actions.append(action)
    episode_states.append(state)

    if state.did_crash:
        episode = [
            train.Experience(
                st, at, stt.distance-st.distance, stt,
            ) for st, at, stt in zip(episode_states[:-1], episode_actions[:-1], episode_states[1:])
        ]
        log("Updating using %i experiences" % len(episode))
        if TRAIN:
            loss, tr = model.update(episode)
            train_data.loss.append(loss)
            train_data.rewards.append(tr)
            train_data.dist.append(episode[-1].stt.distance)
            train_data.timesteps.append(len(episode_actions)+1)
        episode_states = list()
        episode_actions = list()
        num_episodes += 1
        if num_episodes % 100 == 0 and TRAIN:
            log("Saving data")
            with open("model-%s.pkl" % PORT, "wb") as f:
                pickle.dump(model, f)
            train_data.save("autobahn-training-%s" % PORT)
    #log(", ".join(f"{o.obstacle_type}[{round(o.distance)}@{round(o.angle/np.pi*180)}]" for o in train.identify_obstacles(state.sensors)))

    return PredictResponse(
       action=action
    )

if __name__ == "__main__":
    log.configure(
        "racing-cars.log",
        "Racing cars API",
        log_commit=True,
        append=False,
    )
    app.run(host="0.0.0.0", port=PORT, debug=not "Tue")
