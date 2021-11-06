from __future__ import annotations
import subprocess
from dataclasses import dataclass
from functools import wraps
from typing import Any
from argparse import ArgumentParser
import datetime
import json
import logging
import pprint
import time
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from pelutils import log, DataStorage
from pydantic import BaseModel

import train
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

start_time = time.time()
app = Flask(__name__)
Api(app)
CORS(app)

# if TRAIN:
#     model = train.DeepQ(train=True)
# else:
#     with open("model-%s.pkl" % PORT) as f:
#         model = pickle.load(f)
num_episodes = 0
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

state: s.State
@app.route("/api/predict", methods=["POST"])
@api_fun
def predict_train():
    global episode_actions, episode_states, model, num_episodes, prev_info, state
    data = _get_data()

    info = s.Information.from_dict(data)
    episode_states
    if not episode_actions:
        state = s.State(s.Vector(0., 0.), 425, list(), info)
        action = s.ActionType.ACCELERATE
        episode_actions.append(action)
    else:
        action = s.ActionType.NOTHING
        state = state.new_state(info)
        log(pprint.pformat(s.to_dict(state)))
    if info.did_crash:
        episode_actions.clear()
    prev_info = info
    # else:
    #     model.eval()
    #     action = model.predict(episode_states[-1], state)
    #     if TRAIN:
    #         model.train()
    # episode_actions.append(action)
    # episode_states.append(state)

    # if state.did_crash:
    #     episode = [
    #         train.Experience(
    #             st, at, stt.distance-st.distance, stt,
    #         ) for st, at, stt in zip(episode_states[:-1], episode_actions[:-1], episode_states[1:])
    #     ]
    #     log("Updating using %i experiences" % len(episode))
    #     if TRAIN:
    #         loss, tr = model.update(episode)
    #         train_data.loss.append(loss)
    #         train_data.rewards.append(tr)
    #         train_data.dist.append(episode[-1].stt.distance)
    #         train_data.timesteps.append(len(episode_actions)+1)
    #     episode_states = list()
    #     episode_actions = list()
    #     num_episodes += 1
    #     if num_episodes % 100 == 0 and TRAIN:
    #         log("Saving data")
    #         with open("model-%s.pkl" % PORT, "wb") as f:
    #             pickle.dump(model, f)
    #         train_data.save("autobahn-training-%s" % PORT)
    #log(", ".join(f"{o.obstacle_type}[{round(o.distance)}@{round(o.angle/np.pi*180)}]" for o in train.identify_obstacles(state.sensors)))

    return PredictResponse(action=action)

if __name__ == "__main__":
    log.configure(
        f"racing-cars{PORT}.log",
        "Racing cars API",
        log_commit=True,
        append=False,
    )
    app.run(host="0.0.0.0", port=PORT, debug=not "Tue")
