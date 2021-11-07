from __future__ import annotations
from math import sqrt
from typing import Literal
import time

from pelutils import log

import state as s


def to_target_position(state: s.State, target_position: float) -> s.ActionType:

    delta = target_position - state.position
    if delta > 0:
        further_steer = s.ActionType.STEER_RIGHT
        correct_steer = s.ActionType.STEER_LEFT
    elif delta < 0:
        further_steer = s.ActionType.STEER_LEFT
        correct_steer = s.ActionType.STEER_RIGHT
    if delta * state.velocity.y < 0:
        if delta > 0:
            return s.ActionType.STEER_RIGHT
        return s.ActionType.STEER_LEFT

    delta = abs(delta)
    vel = abs(state.velocity.y)
    dist_if_brake = (vel-1) ** 2 / 2 + vel / 2 - 1 / 2
    if dist_if_brake + 2 * vel + 1 < delta:
        return further_steer
    else:
        return correct_steer

# DONE Overstying fra lane 0 til 2
# Skift bane, så snart front-sensor bliver ikke-None
# DONE Styre ind på midten, hvis alle baner er frie
# Hvis yderst, og både samme bane og midterst er optaget, så tænk dig om
# (Niceness) Hvis tid nok, så skiftevis accelerer frem og til siden ved vognbaneskift
# DONE Vognbaneskift skal tage højde for eksisterende fart
# DONE Overvej, om handlingskø bør overskrives

def target_lane(state: s.State, clear_lanes: list[int]) -> int:
    if 1 in clear_lanes:
        return 1
    if state.lane in clear_lanes:
        return state.lane
    if state.lane == 0:
        if 1 in clear_lanes:
            return 1
        return 2
    if state.lane == 1:
        if 0 in clear_lanes:
            return 0
        return 2
    if state.lane == 2:
        if 1 in clear_lanes:
            return 1
        return 0

def predict(state: s.State) -> s.ActionType:
    # First determine what lanes are clear
    # A lane is considered clear if the car can move there without having a car in front
    clear_lanes = list()
    for lane in range(3):
        if not any(
            car.lane == lane and
            ((car.position >= -s.CAR_LENGTH/2 and car.velocity <= 0) or
            (car.position < -s.CAR_LENGTH/2 and car.velocity > 0)) for car in state.cars
        ):
            clear_lanes.append(lane)

    # print(state.position)
    # If our lane is not clear, we have to change
    t_lane = target_lane(state, clear_lanes)
    margin = 30
    if t_lane != state.lane or\
        abs(state.velocity.y) > 0.001 or\
        abs(state.position - s.lane_to_pos(t_lane)) > margin:
        return to_target_position(state, s.lane_to_pos(t_lane, driver=True))

    # If our lane is clear, YEET
    return s.ActionType.ACCELERATE

