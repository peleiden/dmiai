from __future__ import annotations
from math import sqrt
from typing import Literal
import time

from pelutils import log

import state as s


def to_lane_actions(state: s.State, target_lane: Literal[0, 1, 2]) -> list[s.ActionType]:
    # TODO Be as close to the inner line as possible

    delta = s.lane_to_pos(target_lane, driver=True) - state.position
    if delta > 0:
        acc_action = s.ActionType.STEER_RIGHT
        brake_action = s.ActionType.STEER_LEFT
    elif delta < 0:
        acc_action = s.ActionType.STEER_LEFT
        brake_action = s.ActionType.STEER_RIGHT
    else:
        return list()

    delta = abs(delta)
    num_acc_actions = round(sqrt(delta))
    return [acc_action] * num_acc_actions + [brake_action] * num_acc_actions



def predict(state: s.State) -> list[s.ActionType]:
    # TODO Condition lane change if a speedup is required

    # First determine what lanes are clear
    # A lane is considered clear if the car can move there without having a car in front
    clear_lanes = list()
    for lane in range(3):
        if not any(
            car.lane == lane and
            car.position >= -s.CAR_LENGTH/2 and
            car.velocity <= 0 for car in state.cars
        ):
            clear_lanes.append(lane)

    # If our lane is not clear, we have to change
    if state.lane not in clear_lanes:
        if len(clear_lanes) == 1 or 1 not in clear_lanes:
            return to_lane_actions(state, clear_lanes[0])
        else:
            return to_lane_actions(state, 1)

    # If our lane is clear, YEET
    return [s.ActionType.ACCELERATE]

