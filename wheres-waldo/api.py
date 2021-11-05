from __future__ import annotations
from functools import wraps
import datetime
import json
import time
from io import BytesIO
import numpy as np

from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from pelutils import log
from pydantic import BaseModel

from PIL import Image

from yolo_model import setup_model, yolo_predict

model = setup_model()

start_time = time.time()

app = Flask(__name__)
Api(app)
CORS(app)

class PredictResponse(BaseModel):
    x: int
    y: int

def get_uptime() -> str:
    return '{}'.format(datetime.timedelta(seconds=time.time() - start_time))

def _get_data():
    """ Returns the Waldo image """
    return np.array(Image.open(BytesIO(request.file.read())).convert("RGB"))

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
        "service": "wheres-waldo",
    }

@app.route("/api/predict", methods=["POST"])
@api_fun
def predict():
    img = _get_data()
    x, y = yolo_predict(model, img)
    return PredictResponse(x=x, y=y)

if __name__ == "__main__":
    log.configure(
        "wherswaldo.log",
        "Wheres Waldo API",
        log_commit=True,
        append=True,
    )
    app.run(host="0.0.0.0", port=6972, debug=False)
