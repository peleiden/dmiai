from __future__ import annotations
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import os
import random

from pelutils import log
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionType(str, Enum):
    ACCELERATE = 'ACCELERATE'
    DECELERATE = 'DECELERATE'
    STEER_RIGHT = 'STEER_RIGHT'
    STEER_LEFT = 'STEER_LEFT'
    NOTHING = 'NOTHING'

@dataclass
class Velocity:
    x: int
    y: int

@dataclass
class Sensors:
    left_side: float
    left_front: float
    front: float
    right_front: float
    right_side: float
    right_back: float
    back: float
    left_back: float

@dataclass
class State:
    elapsed_time_ms: float
    velocity: Velocity
    sensors: Sensors
    did_crash: bool
    distance: int

    @staticmethod
    def from_dict(d: dict) -> State:
        state = State(**d)
        state.velocity = Velocity(**state.velocity)
        state.sensors = Sensors(**state.sensors)
        for key, value in vars(state.sensors).items():
            if value is None:
                setattr(state.sensors, key, 1000)
        return state

    @property
    def features(self) -> torch.FloatTensor:
        """ Returns the relevant features as an array """
        return torch.FloatTensor([
            self.velocity.x, self.velocity.y, self.sensors.back, self.sensors.front,
            self.sensors.left_back, self.sensors.left_front, self.sensors.left_side,
            self.sensors.right_back, self.sensors.right_front, self.sensors.right_side,
        ])

@dataclass
class Experience:
    st: State
    at: ActionType
    rt: float
    stt: State

FEATURES = 10
ALL_ACTIONS = tuple(a for a in ActionType)
ACTIONS = len(ALL_ACTIONS)

class Model(ABC):
    @abstractmethod
    def __init__(self, train: bool):
        self.train = train

    @abstractmethod
    def predict(self, state: State) -> ActionType:
        pass

    @abstractmethod
    def update(self, episode: list[Experience]):
        pass

class _DeepModel(nn.Module):
    def __init__(self, n_hidden=16):
        super().__init__()
        self.hidden = n_hidden
        self.activation = F.gelu
        self.linear1 = nn.Linear(FEATURES, self.hidden)
        self.dropout = nn.Dropout(p=0.05)
        self.linear2 = nn.Linear(self.hidden, ACTIONS)
        # self.linear3 = nn.Linear(self.hidden, ACTIONS)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class DeepQ(Model):
    def __init__(self, train=False, eps=0.05, gamma=0.99, lr=1e-4, tau=0.1):
        super().__init__(train)
        self.eps = eps
        self.gamma = gamma
        self.lr = lr
        self.tau = tau

        self.pred_network = _DeepModel()
        self.target_network = _DeepModel()

        self._update_target_network(1)
        self.optimizer = torch.optim.AdamW(self.pred_network.parameters())

        if not self.train:
            self.pred_network.eval()
        self.target_network.eval()

    def _update_target_network(self, tau: float):
        pred_sd = self.pred_network.state_dict()
        target_sd = self.target_network.state_dict().copy()
        for key in pred_sd.keys():
            target_sd[key] = tau * pred_sd[key].clone() + (1 - tau) * target_sd[key].clone()
        self.target_network.load_state_dict(target_sd)

    def _Q(self, features: torch.FloatTensor) -> torch.FloatTensor:
        return self.pred_network(features)

    def _Qp(self, features: torch.FloatTensor) -> torch.FloatTensor:
        return self.target_network(features)

    def _predict(self, state: State) -> int:
        if random.random() < self.eps:
            return random.randint(0, ACTIONS-1)
        with torch.no_grad():
            return torch.argmax(self._Q(state.features)).item()

    def predict(self, state) -> ActionType:
        if self.train:
            return ALL_ACTIONS[self._predict(state)]
        with torch.no_grad:
            return ALL_ACTIONS[torch.argmax(self._Q(state)).item()]

    def update(self, episode: list[Experience]):
        x = torch.stack([exp.st.features for exp in episode] + [episode[-1].stt.features])
        rewards = torch.FloatTensor([exp.rt for exp in episode])
        rewards[-1] = -10
        tr = rewards.sum().item()
        actions = torch.LongTensor([ALL_ACTIONS.index(exp.at) for exp in episode])
        with torch.no_grad():
            Qp, _ = self._Qp(x[1:]).max(dim=1)
        Q = self._Q(x[:-1])[torch.arange(len(episode)), actions]
        loss = torch.sum((rewards + self.gamma * Qp - Q) ** 2)
        log("Episode length: %i, total reward: %.4f" % (len(episode), tr))
        loss.backward()
        self.optimizer.step()
        self._update_target_network(self.tau)

        return loss.item(), tr

