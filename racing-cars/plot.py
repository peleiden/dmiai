from api import Data
import numpy as np
import matplotlib.pyplot as plt


data = Data.load("autobahn-training-6971")
plt.plot(data.rewards)
plt.yscale("log")
plt.show()
