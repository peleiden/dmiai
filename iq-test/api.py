from __future__ import annotations
from functools import wraps
import base64
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


from model import ConsistentTransformations

start_time = time.time()
model = ConsistentTransformations()
model2 = HighestSimilarity()

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
    encoded_imgs = [data["image_base64"], *data["image_choices_base64"]]
    imgs = list()
    for img in encoded_imgs:
        img = base64.b64decode(img)
        sbuf = BytesIO()
        sbuf.write(img)
        img = Image.open(sbuf)
        imgs.append(np.array(img))
    #rule_img = np.array(Image.frombytes("RGB", (550, 440), imgs[0]))
    #choice_imgs = [np.array(Image.frombytes("RGB", (110, 110), img)) for img in imgs[1:]]
    return imgs[0], imgs[1:]

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
    rule_img, choice_imgs = _get_data()

    idx = model.predict(rule_img, choice_imgs)
    if idx is None:
        idx = model2.predict(rule_img, choice_imgs)

    log(f"Predicted {idx}")

    return PredictResponse(next_image_index=idx)

if __name__ == "__main__":
    log.configure(
        "iqtest.log",
        "IQ test API",
        log_commit=True,
        append=True,
    )
    app.run(host="0.0.0.0", port=6972, debug=False)
