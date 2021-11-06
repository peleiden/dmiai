from __future__ import annotations
from dataclasses import dataclass
from random import shuffle
import csv
import os

from pelutils import log, Levels, DataStorage
from pelutils.ds.plot import update_rc_params, rc_params_small, figsize_std, figsize_wide, tab_colours
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaModel, AutoConfig, get_linear_schedule_with_warmup, AdamW
import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# Set to None to use max length
NUM_TOKENS = 250
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
        id_arr = torch.full((NUM_TOKENS,), fill_value=tokenizer.pad_token_id, dtype=torch.int32)
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(review.review))
        ids = ids[:NUM_TOKENS-2]
        id_arr[0] = tokenizer.cls_token_id
        id_arr[1:1+len(ids)] = torch.IntTensor(ids)
        id_arr[1+len(ids)] = tokenizer.sep_token_id
        return Example(
            id_arr,
            parse_score(review),
        )

@dataclass
class Batch:
    ids: torch.IntTensor  # batch size x num ids
    targets: torch.LongTensor  # batch size

    midpoints = np.linspace(-0.5, 0.5, 11)
    midpoints = (midpoints[:-1]+midpoints[1:]) / 2
    stars = np.linspace(0.5, 5, 10)

    @classmethod
    def from_examples(cls, examples: list[Example]) -> Batch:
        targets = torch.argmin(torch.abs(torch.vstack([torch.Tensor([ex.target for ex in examples])]*10).T-cls.midpoints), dim=1).long()
        return Batch(
            torch.stack([ex.ids for ex in examples]),
            targets,
        )

    def to(self, device: torch.device) -> Batch:
        return Batch(
            self.ids.to(device),
            self.targets.to(device),
        )

    def __len__(self) -> int:
        return len(self.targets)

@dataclass
class Results(DataStorage):
    epochs: int
    num_batches: int

    train_losses: np.ndarray
    test_losses: np.ndarray
    accuracies: np.ndarray

class ScorePredictor(nn.Module):

    def __init__(self, pretrained_model: nn.Module, bert_config: AutoConfig):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Linear(bert_config.hidden_size*NUM_TOKENS, len(Batch.stars))

    def forward(self, batch: Batch) -> torch.FloatTensor:
        cwrs = self.pretrained_model(batch.ids, return_dict=True)["last_hidden_state"]
        cwrs = cwrs.view(len(batch), -1).contiguous()
        return self.classifier(cwrs).squeeze()

def parse_score(review: Review) -> float:
    """ Parses a review score. Returns -1 if unparsable """
    if review.score.endswith("/5"):
        nom, denom = review.score.split("/")
        if denom not in ("0", "00"):
            score = float(nom) / float(denom)
            if 0 <= score <= 1:
                return score - 0.5
    return -1

def coerce_scores(scores: torch.FloatTensor) -> torch.FloatTensor:
    """ Converts predictions in [0, 1] to the used interval [0.5, 5] """
    scores = scores.clone()
    scores += 0.5
    scores *= 5
    scores[scores<0.5] = 0.5
    scores[scores>5] = 5
    return scores

def load_data(max_examples: int, test_train_split=0.01) -> tuple[list[Review], list[Review]]:
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

def evaluate(model, tokenizer, num_test_batches: int, test_batches: list[Batch], device: torch.device, criterion, res: Results, i: int, plot_preds=False, location: str=""):
    log("Evaluating model")
    if plot_preds:
        plt.figure(figsize=figsize_wide)
        plt.subplot(121)
        plt.xlabel("Target")
        plt.ylabel("Predicted")
        plt.plot([-.5, .5], [-.5, .5])
        plt.title("Training domain")
        plt.grid()
        plt.subplot(122)
        plt.xlabel("Target")
        plt.ylabel("Predicted")
        plt.plot([0, 5], [0, 5])
        plt.title("Prediction domain")
        plt.subplot(121)
    with torch.no_grad():
        model.eval()
        losses = np.zeros(num_test_batches)
        accuracies = np.zeros(num_test_batches)
        for j, batch in tqdm(enumerate(test_batches), total=num_test_batches):
            batch = batch.to(device)
            out = model(batch)
            losses[j] = criterion(out, batch.targets).item()
            pred_stars = Batch.stars[torch.argmax(out, dim=1).cpu().numpy()]
            accuracies[j] = np.abs(pred_stars-Batch.stars[batch.targets.cpu().numpy()]).mean()
            if plot_preds:
                plt.figure(figsize=figsize_std)
                plt.scatter(Batch.stars[batch.targets.cpu().numpy()], pred_stars, c=tab_colours[0])
        res.test_losses[i+1] = losses.mean()
        res.accuracies[i+1] = accuracies.mean()
        log("Mean test loss: %.4f" % res.test_losses[i+1], "Mean accuracy %.4f" % res.accuracies[i+1])
        model.train()
    if plot_preds:
        plt.tight_layout()
        plt.savefig(os.path.join(location, "preds" + ".png"))
        plt.close()

@click.command()
@click.argument("location")
@click.option("-m", "--model-name", default="roberta-base")
@click.option("-b", "--batch-size", default=8, type=int)
@click.option("--epochs", default=5, type=int)
@click.option("--lr", default=1e-6, type=float)
@click.option("--max-examples", default=0, type=int)
def run(location: str, model_name: str, batch_size: int, epochs: int, lr: float, max_examples: int):

    log.configure(
        os.path.join(location, "movie-reviews-train.log"),
        "Moview reviews training",
        print_level=Levels.DEBUG,
        log_commit=True,
    )
    log("Location:          %s" % location,
        "Model:             %s" % model_name,
        "Batch size:        %i" % batch_size,
        "Epochs:            %i" % epochs,
        "Learning rate:     %s" % lr,
        "Max examples:      %s" % max_examples,
        "Number of  tokens: %i" % NUM_TOKENS)
    def savefig(name: str):
        plt.tight_layout()
        plt.savefig(os.path.join(location, name + ".png"))
        plt.close()

    log.section("Loading data")
    train_reviews, test_reviews = load_data(max_examples)
    log("Loaded %i training reviews and %i test reviews" % (len(train_reviews), len(test_reviews)))

    log.section("Building tokenizer and model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model = ScorePredictor(model, bert_config).to(device)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_updates = num_train_batches * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06*num_updates), num_updates)
    criterion = nn.CrossEntropyLoss()

    log.section("Training for %i epochs for a total of %i updates" % (epochs, num_updates))
    res = Results(
        epochs = epochs,
        num_batches = num_train_batches,
        train_losses = np.zeros(num_updates),
        test_losses = np.zeros(epochs+1),
        accuracies = np.zeros(epochs+1)
    )
    evaluate(model, tokenizer, num_test_batches, test_batches, device, criterion, res, -1)
    for i in range(epochs):
        log("Starting epoch %i" % i)
        shuffle(train_batches)
        for j, batch in enumerate(train_batches):
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.targets)
            log.debug("Batch %i/%i (ep. %i/%i): Loss %.4f" % (j, num_train_batches-1, i, epochs-1, loss.item()))
            res.train_losses[i*num_train_batches+j] = loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        evaluate(model, tokenizer, num_test_batches, test_batches, device, criterion, res, i, i==epochs-1, location)

        log("Saving model")
        torch.save(model.state_dict(), os.path.join(location, model_name+f"_epoch_{i}.pt"))
        res.save(location)

    log.section("Saving results to %s" % location)
    res.save(location)

if __name__ == "__main__":
    with log.log_errors:
        run()
