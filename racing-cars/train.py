from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import numpy as np
import random

from pelutils import log
import torch
import torch.nn as nn
import torch.nn.functional as F

import state


# One hot encode y position of other cars
# Give velocities by lane

WIDTH = 425+450
CAR_WIDTH = 425-200.75
MARGIN = 10
NONE_PLACEHOLDER = 1000
RIGHT_DIAGONAL_NONE = 1133
LEFT_DIAGONAL_NONE = 1097

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
        for diagonal_attr in ("left_back", "left_front", "right_back", "right_front"):
            if getattr(state.sensors, diagonal_attr) is None:
                setattr(state.sensors, diagonal_attr, LEFT_DIAGONAL_NONE if "left" in diagonal_attr else RIGHT_DIAGONAL_NONE)
        for straight_attr in ("left_side", "right_side"):
            if getattr(state.sensors, straight_attr) is None:
                setattr(state.sensors, straight_attr, 0)
        return state

    @property
    def features(self) -> torch.FloatTensor:
        """ Returns the relevant features as an array """
        return torch.FloatTensor([
            f if f is not None else NONE_PLACEHOLDER for f in (
                self.velocity.x,
                self.velocity.y,
                self.sensors.back,
                self.sensors.front,
                self.sensors.left_back,
                self.sensors.left_front,
                self.sensors.left_side,
                self.sensors.right_back,
                self.sensors.right_front,
                self.sensors.right_side,
            )
        ])

    @property
    def features(self) -> torch.FloatTensor:
        """ Returns the relevant features as an array """
        obstacle_features = build_distance_features(identify_obstacles(self.sensors))

        return torch.FloatTensor([
            f if f is not None else NONE_PLACEHOLDER for f in (
                self.velocity.x,
                self.velocity.y,
                *obstacle_features,
            )
        ])

@dataclass
class Experience:
    st: state.State
    at: ActionType
    rt: float
    stt: state.State

FEATURES = 13
ALL_ACTIONS = tuple(a for a in ActionType)
ACTIONS = len(ALL_ACTIONS)

class Model(ABC):
    @abstractmethod
    def __init__(self, train: bool):
        self._train = train

    @abstractmethod
    def predict(self, *states: State) -> ActionType:
        pass

    @abstractmethod
    def update(self, episode: list[Experience]):
        pass

class _DeepModel(nn.Module):
    def __init__(self, n_hidden=256):
        super().__init__()
        self.hidden = n_hidden
        self.activation = F.gelu
        self.dropout = nn.Dropout(p=0.05)
        self.linear1 = nn.Linear(FEATURES, self.hidden)
        self.linear2 = nn.Linear(self.hidden, ACTIONS)

    def forward(self, x: torch.FloatTensor):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x.squeeze()

class DeepQ(Model):
    def __init__(self, train=False, eps=0.05, gamma=0.95, lr=1e-4, tau=0.1):
        super().__init__(train)
        self.eps = eps
        self.gamma = gamma
        self.lr = lr
        self.tau = tau

        self.pred_network = _DeepModel()
        self.target_network = _DeepModel()

        self._update_target_network(1)
        self.optimizer = torch.optim.AdamW(self.pred_network.parameters())

        if not self._train:
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

    def _predict(self, state: state.State) -> int:
        if random.random() < self.eps:
            return random.randint(0, ACTIONS-1)
        with torch.no_grad():
            return torch.argmax(self._Q(state.features.unsqueeze(0))[0]).item()

    def predict(self, state: state.State) -> ActionType:
        if self._train:
            return ALL_ACTIONS[self._predict(state)]
        with torch.no_grad:
            return ALL_ACTIONS[torch.argmax(self._Q(state.features.unsqueeze(0))[0])]

    def update(self, episode: list[Experience]):
        # Features consist of current and previous state to model velocity
        x = torch.vstack([ex.st.features for ex in episode] + [episode[-1].stt.features])
        rewards = torch.FloatTensor([exp.rt for exp in episode])
        tr = rewards.sum().item()
        actions = torch.LongTensor([ALL_ACTIONS.index(exp.at) for exp in episode])
        # breakpoint()
        with torch.no_grad():
            Qp, _ = self._Qp(x[1:]).max(dim=1)
        Q = self._Q(x[:-1])[torch.arange(len(episode)), actions]
        loss = torch.sum((rewards + self.gamma * Qp - Q) ** 2)
        log("Episode length: %i, total reward: %.4f" % (len(episode), tr))
        loss.backward()
        self.optimizer.step()
        self.pred_network.zero_grad()

        self._update_target_network(self.tau)

        return loss.item(), tr

    def train(self):
        self.pred_network.train()

    def eval(self):
        self.pred_network.eval()

@dataclass
class Obstacle:
    obstacle_type: str # "car" or "wall"
    distance: float
    angle: float

def diagonal(straight_dist: float) -> float:
    return straight_dist/np.cos(np.pi/4)

def straight(diag_dist: float) -> float:
    return diag_dist*np.cos(np.pi/4)

def wall_consistent(exp: float,  back: float, front: float):
    return (exp - MARGIN < back < exp + MARGIN) and (exp - MARGIN < front < exp + MARGIN)

def identify_obstacles(sensors: Sensors) -> list[Obstacle]:
    obstacles = list()
    # Expected diagonals if at wall
    exp_l, exp_r = diagonal(sensors.left_side), diagonal(sensors.right_side)

    # Handle cars in front of/behind us
    for a, s in zip((0, np.pi), (sensors.front, sensors.back)):
        if s is not None:
            obstacles.append(Obstacle("car", s, a))

    # Handle obstacles left/right
    car_side = []
    if sensors.left_side + sensors.right_side < (WIDTH - MARGIN): # Then, there is at least one car
        car_side = list("lr")
        # Check whether the side distances are consistent with being at a wall
        # NOTE: We assume that it is impossible to have multiple cars within same side detectors
        if wall_consistent(exp_l, sensors.left_back, sensors.left_front):
            obstacles.append(Obstacle("wall", sensors.left_side, 1/2*np.pi))
            car_side.remove("l")
        if wall_consistent(exp_r, sensors.right_back, sensors.right_front):
            obstacles.append(Obstacle("wall", sensors.right_side, 3/2*np.pi))
            car_side.remove("r")
        if "l" in car_side:
            obstacles.append(Obstacle("car", sensors.left_side, 1/2*np.pi))
            obstacles.append(Obstacle("wall", straight(max(sensors.left_back, sensors.left_front)) , 1/2*np.pi))
        if "r" in car_side:
            obstacles.append(Obstacle("car", sensors.right_side, 3/2*np.pi))
            obstacles.append(Obstacle("wall", straight(max(sensors.right_back, sensors.right_front)), 3/2*np.pi))
    else:
        obstacles.append(Obstacle("wall", sensors.left_side, 1/2*np.pi))
        obstacles.append(Obstacle("wall", sensors.right_side, 3/2*np.pi))

    # Check for cars on our diagonals.
    # NOTE: We only check diagonals where there is no car to avod spotting car multiple times
    if not "l" in car_side:
        if sensors.left_front < (exp_l - MARGIN):
            obstacles.append(Obstacle("car", sensors.left_front, 1/4*np.pi))
        if sensors.left_back < (exp_l - MARGIN):
            obstacles.append(Obstacle("car", sensors.left_back, 3/4*np.pi))
    if not "r" in car_side:
        if sensors.right_front < (exp_r - MARGIN): # Margin
            obstacles.append(Obstacle("car", sensors.right_front, 7/4*np.pi))
        if sensors.right_back < (exp_r - MARGIN):
            obstacles.append(Obstacle("car", sensors.right_back, 5/4*np.pi))
    return obstacles

def build_distance_features(obstacles: list[None | Obstacle]) -> list[float]:
    features = [None]*6
    for o in obstacles:
        if o.obstacle_type == "wall":
            if o.angle == 1/2*np.pi:
                features[0] = o.distance
            elif o.angle == 3/2*np.pi:
                features[1] = o.distance
            else:
                raise ValueError("Wall angle?")
        elif o.obstacle_type == "car":
            if o.angle == 0:
                features[2] = o.distance
            elif o.angle == np.pi:
                features[3] = o.distance
            elif o.angle < np.pi:
                features[4] = o.distance
            else:
                features[5] = o.distance
    return features
