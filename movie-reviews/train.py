from __future__ import annotations
from dataclasses import dataclass
from random import shuffle
import csv
import os

from pelutils import log, Levels, DataStorage
from pelutils.ds.plot import update_rc_params, rc_params_small, figsize_std
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaModel, AutoConfig, get_linear_schedule_with_warmup, AdamW
import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

OUTPUT = "output"
os.makedirs(OUTPUT, exist_ok=True)


update_rc_params(rc_params_small)

@dataclass
class Review:
    review: str
    score: str

@dataclass
class Example:
    ids: torch.IntTensor
    target: float  # From 0 to 1

    @staticmethod
    def from_review(tokenizer, review: Review) -> Example:
        id_arr = torch.zeros(tokenizer.model_max_length, dtype=torch.int32)
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(review.review))[:tokenizer.model_max_length-2]
        id_arr[0] = tokenizer.cls_token_id
        id_arr[1:1+len(ids)] = torch.IntTensor(ids)
        id_arr[1+len(ids)] = tokenizer.sep_token_id
        return Example(
            id_arr,
            fix_score(review),
        )

@dataclass
class Batch:
    ids: torch.IntTensor  # batch size x max length
    targets: torch.FloatTensor  # batch size

    @staticmethod
    def from_examples(examples: list[Example]) -> Batch:
        return Batch(
            torch.stack([ex.ids for ex in examples]),
            torch.FloatTensor([ex.target for ex in examples])
        )

    def to(self, device: torch.device) -> Batch:
        return Batch(
            self.ids.to(device),
            self.targets.to(device)
        )

class ScorePredictor(nn.Module):

    def __init__(self, pretrained_model: nn.Module, bert_config: AutoConfig):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.regressor = nn.Linear(bert_config.hidden_size, 1)

    def forward(self, batch: Batch) -> torch.FloatTensor:
        cwrs = self.pretrained_model(batch.ids, return_dict=True)
        return self.regressor(cwrs["last_hidden_state"])

def fix_score(review: Review) -> float:
    if "/" in review.score:
        nom, denom = review.score.split("/")
        if "0" not in (nom, denom) and "00" not in (nom, denom):
            score = float(nom) / float(denom)
            if 0 < score <= 5:
                return score
    return -1

def load_data(max_examples: int, test_train_split=0.1) -> tuple[list[Review], list[Review]]:
    reviews = list()
    with open("data/rotten_tomatoes_critic_reviews.csv") as f:
        r = csv.reader(f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
        for i, row in enumerate(tqdm(r)):
            if max_examples and i == max_examples:
                break
            reviews.append(Review(row[-1], row[5]))

    shuffle(reviews)
    n_train = int((1-test_train_split) * len(reviews))
    return reviews[:n_train], reviews[n_train:]

def savefig(name: str):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, name + ".png"))
    plt.close()

@click.command()
@click.option("-m", "--model-name", default="roberta-base")
@click.option("-b", "--batch-size", default=8, type=int)
@click.option("--epochs", default=5, type=int)
@click.option("--lr", default=1e-4, type=float)
@click.option("--max-examples", default=0, type=int)
def run(model_name: str, batch_size: int, epochs: int, lr: float, max_examples: int):
    log.section("Loading data")
    train_reviews, test_reviews = load_data(max_examples)
    log("Loaded %i training reviews and %i test reviews" % (len(train_reviews), len(test_reviews)))

    log.section("Building tokenizer and model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_config = AutoConfig.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model = ScorePredictor(model, bert_config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    log.section("Building examples")
    train_examples = [ex for review in tqdm(train_reviews) if (ex := Example.from_review(tokenizer, review)).target != -1]
    test_examples = [ex for review in tqdm(test_reviews) if (ex := Example.from_review(tokenizer, review)).target != -1]
    log("Has %i training examples and %i test examples" % (len(train_examples), len(test_examples)))

    train_targets = np.array([ex.target for ex in train_examples])
    plt.figure(figsize=figsize_std)
    plt.hist(train_targets, bins=50)
    plt.title("Score distribution in training data\nMean = %.2f, std = %.2f" % (train_targets.mean(), train_targets.std()))

    savefig("data-dist")

    # Creating batches
    log.section("Batching data")
    num_train_batches = len(train_examples) // batch_size
    train_batches = [Batch.from_examples(train_examples[i*batch_size:(i+1)*batch_size]) for i in tqdm(range(num_train_batches))]
    num_test_batches = len(test_examples) // batch_size
    test_batches = [Batch.from_examples(test_examples[i*batch_size:(i+1)*batch_size]) for i in tqdm(range(num_test_batches))]
    log("Created %i training batches and %i test batches" % (num_train_batches, num_test_batches))

    log.section("Setting up loss, scheduler, and optimizer")
    optimizer = AdamW([{"params": [p for _, p in model.named_parameters()], "weight_decay": 0.01}], lr=lr)
    num_updates = num_train_batches * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06*num_updates), num_updates)
    criterion = nn.L1Loss(reduction="sum")

    log.section("Training for %i epochs for a total of %i updates" % (epochs, num_train_batches))
    for i in range(epochs):
        log("Starting epoch %i" % i)
        shuffle(train_batches)
        for j, batch in enumerate(train_batches):
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.targets)
            log.debug("Batch %i (ep. %i): Loss %.4f" % (j, i, loss.item()))
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()



if __name__ == "__main__":
    log.configure("movie-reviews-train.log", "Moview reviews training", print_level=Levels.DEBUG, log_commit=True)
    with log.log_errors:
        run()
