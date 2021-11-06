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
import gym

class ActionType(str, Enum):
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'

@dataclass
class State:
    cart_pos: float
    cart_vel: float
    pole_angl: float
    pole_angl_vel: float

    @property
    def features(self) -> torch.FloatTensor:
        """ Returns the relevant features as an array """
        return torch.FloatTensor([
            self.cart_pos,
            self.cart_vel,
            self.pole_angl,
            self.pole_angl_vel
        ])

@dataclass
class Experience:
    st: State
    at: ActionType
    rt: float
    stt: State

FEATURES = 4
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
    def __init__(self, n_hidden=64):
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
        tr = rewards.sum().item()
        actions = torch.LongTensor([ALL_ACTIONS.index(exp.at) for exp in episode])
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

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    model = DeepQ(train=True)
    num_episodes = 0
    episode_actions = list()
    episode_states = list()
    episode_rewards = list()
    while True:
        state = State(*env.reset())
        while True:
            action = model.predict(state)
            episode_actions.append(action)
            episode_states.append(state)


            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            episode_rewards.append(reward)
            if terminal:
                episode = [
                    Experience(st, at, rt) for st, at, rt in zip(episode_states, episode_actions, episode_rewards)
                ]
                log("Updating using %i experiences" % len(episode))
                if TRAIN:
                    loss, tr = model.update(episode)
                    train_data.loss.append(loss)
                    train_data.rewards.append(tr)
                    train_data.dist.append(episode[-1].stt.distance)
                    train_data.timesteps.append(len(episode_actions)+1)
                episode_states = list()
                episode_actions = list()
                num_episodes += 1
                if num_episodes % 100 == 0 and TRAIN:
                    log("Saving data")
                    with open("model-%s.pkl" % PORT, "wb") as f:
                        pickle.dump(model, f)
                    train_data.save("autobahn-training-%s" % PORT)
            #log(", ".join(f"{o.obstacle_type}[{round(o.distance)}@{round(o.angle/np.pi*180)}]" for o in train.identify_obstacles(state.sensors)))

            return PredictResponse(
            action=action
            )
