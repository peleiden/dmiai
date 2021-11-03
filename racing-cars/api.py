from __future__ import annotations
from enum import Enum
from functools import wraps
from typing import Any, List
import datetime
import json
import logging
import time
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from pelutils import log
from pydantic import BaseModel
import numpy as np

import train

start_time = time.time()
app = Flask(__name__)
Api(app)
CORS(app)

model = train.DeepQ()
episode_actions = list()
episode_states = list()

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
    global episode_actions, episode_states, model
    data = _get_data()

    state = train.State.from_dict(data)
    action = model.predict(state, train=True)
    # log("Making action %s" % action)
    episode_actions.append(action)

    episode_states.append(state)
    if state.did_crash:
        episodes = [
            train.Experience(
                st, at, stt.distance-st.distance, stt,
            ) for st, at, stt in zip(episode_states[:-1], episode_actions[:-1], episode_states[1:])
        ]
        log("Updating using %i experiences" % len(episodes))
        model.update(episodes)
        episode_states = list()
        episode_actions = list()

    import random
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
    app.run(host="0.0.0.0", port=6971, debug="Tue")
