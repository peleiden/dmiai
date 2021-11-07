import os
import shutil

from pelutils.ds.plot import update_rc_params, rc_params_small, figsize_std
import click
import matplotlib.pyplot as plt
import numpy as np

from train import Results

loc: str
update_rc_params(rc_params_small)

def savefig(name: str):
    plt.tight_layout()
    plt.savefig(os.path.join(loc, name))
    plt.close()

def loss_and_acc(res: Results):
    plt.figure(figsize=figsize_std)
    pus = np.arange(res.epochs*res.num_batches)
    epochs = np.arange(1+res.epochs) * res.num_batches
    plt.plot(pus, res.train_losses, label="Train loss")
    plt.scatter(epochs, res.test_losses, label="Test loss", edgecolors="black")
    plt.scatter(epochs, res.errors, label="Accuracy", edgecolors="black")
    plt.legend(loc=1)
    plt.grid()

    savefig("loss-and-accuracy")

@click.command()
@click.argument("location")
def run(location: str):
    global loc
    loc = os.path.join(location, "plots")
    shutil.rmtree(loc, ignore_errors=True)
    os.makedirs(loc)

    res = Results.load(location)

    loss_and_acc(res)

if __name__ == "__main__":
    run()
