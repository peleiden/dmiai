from __future__ import annotations
from dataclasses import dataclass
from random import shuffle
import csv

from pelutils import log, Levels
import torch

@dataclass
class Review:
    review: str
    score: str

@dataclass
class Example:
    ids: torch.IntTensor
    target: float  # From 0 to 1

def load_data(test_train_split=0.1) -> tuple[list[Review], list[Review]]:
    reviews = list()
    with open("data/rotten_tomatoes_critic_reviews.csv") as f:
        r = csv.reader(f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
        for row in r:
            reviews.append(Review(row[-1], row[5]))

    shuffle(reviews)
    n_train = (1-test_train_split) * len(reviews)
    return reviews[:n_train], reviews[n_train:]

def fix_score(review: Review) -> float:
    if "/" in review.score:
        nom, denom = review.score.split("/")
        if nom != "0":
            return float(nom) / float(denom)

def run():
    log.section("Loading data")
    train_reviews, test_reviews = load_data()
    log("Loaded %i training reviews and %i test reviews" % (len(train_reviews), len(test_reviews)))

    log.section("Building examples")
    # TODO
    # Throw out data based on number of unks (crude way of filtering non-english)
    # Filter scores, maybe with harder
    # Batch and train


if __name__ == "__main__":
    log.configure("movie-reviews-train.log", "Moview reviews training", print_level=Levels.DEBUG, log_commit=True)
    with log.log_errors:
        run()
