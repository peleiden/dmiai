from __future__ import annotations
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
from transformers import AutoTokenizer, RobertaModel, AutoConfig
import torch

from train import ScorePredictor, coerce_scores, Review, Example, Batch

model_name = "roberta-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)
model = ScorePredictor(model, bert_config)
sd = torch.load("movie-reviews/%s.pt" % model_name, map_location=torch.device("cpu"))
model.load_state_dict(sd)
model = model.to(device)
model.eval()

start_time = time.time()
app = Flask(__name__)
Api(app)
CORS(app)


class PredictResponse(BaseModel):
    ratings: List[float]

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
    review_strings = data["reviews"]
    log.debug("Received %i reviews" % len(review_strings))

    # Score does not matter here, it just has to be parsable for code to work
    reviews = [Review(s, "5/5") for s in review_strings]
    examples = [Example.from_review(tokenizer, review) for review in reviews]

    # Build batches
    batch_size = 100
    num_batches = len(examples) // batch_size
    batches = list()
    for i in range(num_batches):
        batches.append(Batch.from_examples(examples[i*batch_size:(i+1)*batch_size]))
    if (remaining_examples := len(examples) % batch_size) != 0:
        batches.append(Batch.from_examples(examples[-remaining_examples:]))
        num_batches += 1
    assert num_batches == len(batches)

    # Predict number of stars
    preds = torch.empty(len(examples), dtype=float)
    with torch.no_grad(), torch.cuda.amp.autocast_mode.autocast():
        c = 0
        for i, batch in enumerate(batches):
            log.debug("Processing batch %i / %i, which has size %i" % (i, num_batches-1, len(batch)))
            preds[c:c+len(batch)] = model(batch.to(device))
            c += len(batch)
        log.debug("Done predicting scores")
        assert c == len(examples)

    return PredictResponse(ratings=coerce_scores(preds).tolist())

if __name__ == "__main__":
    log.configure(
        "movie-reviews.log",
        "Movie reviews API",
        log_commit=True,
        append=True,
    )
    log("Loaded model %s" % model_name)
    app.run(host="0.0.0.0", port=6969, debug=False)
