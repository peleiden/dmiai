from __future__ import annotations
from functools import wraps
from typing import Any
from argparse import ArgumentParser
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

from model import predict
import state as s


p = ArgumentParser()
p.add_argument("--port", type=int, default=6971)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

PORT = a.port

start_time = time.time()
app = Flask(__name__)
Api(app)
CORS(app)

init = True
state: s.State

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
            log("Received call to %s" % func.__name__)
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

@app.route("/api/predict", methods=["POST"])
@api_fun
def predict():
    global state, init
    data = _get_data()

    info = s.Information.from_dict(data)

    if init:
        state = s.State(0, s.Vector(0., 0.), 425, list(), info)
        action = s.ActionType.ACCELERATE
        init = False
    else:
        state = state.new_state(info)
        action = predict(state)

    if info.did_crash:
        log("Crashed, get better", state.info.distance)
        init = True

    return PredictResponse(action=action)

if __name__ == "__main__":
    log.configure(
        f"racing-cars{PORT}.log",
        "Racing cars API",
        log_commit=True,
        append=False,
    )
    app.run(host="0.0.0.0", port=PORT, debug=not "Tue")
