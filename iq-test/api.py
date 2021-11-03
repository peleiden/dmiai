from __future__ import annotations
from functools import wraps
from typing import Any, List
import base64
import datetime
import json
import time

from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from pelutils import log
from pydantic import BaseModel


start_time = time.time()
app = Flask(__name__)
Api(app)
CORS(app)

class PredictResponse(BaseModel):
    next_image_index: int

def get_uptime() -> str:
    return '{}'.format(datetime.timedelta(seconds=time.time() - start_time))

def _get_data():
    """ Returns the five images for IQ test """
    data = json.loads(request.data.decode("ascii"))
    imgs = list()
    imgs.append(data["image_base64"])
    imgs.extend(data["image_choices_base64"])
    imgs = tuple(base64.b64decode(img) for img in imgs)
    return imgs

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
        "service": "iq-test",
    }

@app.route("/api/predict", methods=["POST"])
@api_fun
def predict():
    data = _get_data()
    return PredictResponse(next_image_index=3)

if __name__ == "__main__":
    log.configure(
        "iqtest.log",
        "IQ test API",
        log_commit=True,
        append=True,
    )
    app.run(host="0.0.0.0", port=6972, debug=False)
