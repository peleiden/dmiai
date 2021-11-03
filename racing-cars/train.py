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
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, state: State, train=False) -> ActionType:
        pass

    @abstractmethod
    def update(self, episode: list[Experience]):
        pass

class DeepQ(Model):
    def __init__(self, eps=0.2, gamma=0.95, lr=1e-4, tau=0.2):
        self.eps = eps
        self.gamma = gamma
        self.lr = lr
        self.tau = tau

        self.pred_network = nn.Linear(FEATURES, ACTIONS)
        self.target_network = nn.Linear(FEATURES, ACTIONS)
        self._update_target_network(1)
        self.optimizer = torch.optim.AdamW(self.pred_network.parameters())

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

    def predict(self, state, train=False) -> ActionType:
        if train:
            return ALL_ACTIONS[self._predict(state)]
        with torch.no_grad:
            return ALL_ACTIONS[torch.argmax(self._Q(state)).item()]

    def update(self, episode: list[Experience]):
        x = torch.stack([exp.st.features for exp in episode] + [episode[-1].stt.features])
        rewards = torch.FloatTensor([exp.rt for exp in episode])
        log(rewards)
        rewards[-1] = -100
        actions = torch.LongTensor([ALL_ACTIONS.index(exp.at) for exp in episode])
        with torch.no_grad():
            Qp, _ = self._Qp(x[1:]).max(dim=1)
        Q = self._Q(x[:-1])[torch.arange(len(episode)), actions]
        loss = torch.mean((rewards + self.gamma * Qp - Q) ** 2)
        log("Episode length: %i, loss: %.4f" % (len(episode), loss.item()))
        loss.backward()
        self.optimizer.step()
        self._update_target_network(self.tau)

