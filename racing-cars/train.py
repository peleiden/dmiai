from __future__ import annotations
from abc import ABC
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
        if state.sensors.back is None:
            state.sensors.back = 2000
        if state.sensors.front is None:
            state.sensors.front = 2000
        return state

    def vector(self) -> torch.FloatTensor:
        """ Returns the relevant features as an array """
        return torch.FloatTensor([
            self.velocity, self.sensors.back, self.sensors.front, self.sensors.left_back,
            self.sensors.left_front, self.sensors.left_side, self.sensors.right_back,
            self.sensors.right_front, self.sensors.right_side,
        ])

@dataclass
class Experience:
    st: State
    at: ActionType
    rt: float
    stt: State

FEATURES = 9
ALL_ACTIONS = tuple(a for a in ActionType)
ACTIONS = len(ALL_ACTIONS)

class Model(ABC):
    def __init__(self):
        pass

    def predict(self, state: State) -> ActionType:
        pass

    def update(self, episode: list[Experience]):
        pass

class DeepQ(Model):
    def __init__(self, eps=0.1):
        self.eps = eps
        self.loss_fn = nn.MSELoss()
        self.network = nn.Linear(FEATURES, ACTIONS)
        self.optimizer = torch.optim.AdamW(self.network.parameters())

    def _Q(self, features: torch.FloatTensor) -> torch.FloatTensor:
        return self.network(features)

    def _greedy_predict(self, state: State) -> int:
        if random.random() < self.eps:
            return random.randint(0, ACTIONS-1)
        with torch.no_grad:
            torch.argmax(self._Q(state)).item()

    def predict(self, state) -> ActionType:
        with torch.no_grad:
            return ALL_ACTIONS[torch.argmax(self._Q(state)).item()]


