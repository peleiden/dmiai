from __future__ import annotations
from enum import Enum
from functools import wraps
from typing import Any, List
import datetime
import json
import time

from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from pelutils import log
from pydantic import BaseModel
import numpy as np

start_time = time.time()
app = Flask(__name__)
Api(app)
CORS(app)

class ActionType(str, Enum):
    ACCELERATE = 'ACCELERATE'
    DECELERATE = 'DECELERATE'
    STEER_RIGHT = 'STEER_RIGHT'
    STEER_LEFT = 'STEER_LEFT'
    NOTHING = 'NOTHING'

class PredictResponse(BaseModel):
    action: ActionType

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
        "service": "movie-reviews",
    }

@app.route("/api/predict", methods=["POST"])
@api_fun
def predict():
    data = _get_data()
    actions = [ActionType.ACCELERATE, ActionType.DECELERATE,
            ActionType.STEER_LEFT, ActionType.STEER_RIGHT,
            ActionType.NOTHING]
    log(data)

    import random
    return PredictResponse(
        action=random.choice(actions)
    )

if __name__ == "__main__":
    log.configure(
        "racing-cars.log",
        "Racing cars API",
        log_commit=True,
        append=False,
    )
    app.run(host="0.0.0.0", port=6971, debug=False)
