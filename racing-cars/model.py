from __future__ import annotations
from math import sqrt
from typing import Literal
import time

from pelutils import log

import state as s


MARGIN = 30

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
# DONE Vognbaneskift skal tage højde for eksisterende fart
# DONE Overvej, om handlingskø bør overskrives
# DONE Styre ind på midten, hvis alle baner er frie
# DONE Skift bane, så snart front-sensor bliver ikke-None
# DONE Hvis yderst, og både samme bane og midterst er optaget, så tænk dig om
# DONE Første state skal beregnes, hvis der ikke er kørt noget
# Lad være at køre for hurtigt, se seed 3
# Tilføj try/except
# (Niceness) Hvis tid nok, så skiftevis accelerer frem og til siden ved vognbaneskift

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
        if 0 not in clear_lanes:
            return 2
        if 2 not in clear_lanes:
            return 0
        if state.position > s.lane_to_pos(state.lane):
            return 2
        return 0
    if state.lane == 2:
        if 1 in clear_lanes:
            return 1
        return 0

def find_clear_lanes(state: s.State) -> list[int]:
    non_clear = set()

    for lane in range(3):
        # If there is a car and it is ahead us and going faster than us or behind us and going slower than us,
        # then the lane is clear
        if (cars := [car for car in state.cars if car.lane == lane]):
            if any(
                # If the car is ahead of us, it must not have negative speed or be too close
                (car.position >= 0 and (car.velocity < 0 or car.position <= s.CAR_LENGTH + 50))
                or
                # If the car is behind us, it must not be too close
                (car.position < 0 and car.position >= -s.CAR_LENGTH - 50)
                    for car in cars
            ):
                non_clear.add(lane)
        elif state.lane == lane and state.info.sensors.front is not None:
            non_clear.add(lane)
        else:
            diff = lane-state.lane
            if diff == 1 and not s.wall_consistent(state, "right"):
                non_clear.add(lane)
            if diff == -1 and not s.wall_consistent(state, "left"):
                non_clear.add(lane)

    return list(set(range(3))-non_clear)

def predict(state: s.State) -> s.ActionType:
    # First determine what lanes are clear
    # A lane is considered clear if the car can move there without having a car in front
    print(state.cars)
    clear_lanes = find_clear_lanes(state)

    t_lane = target_lane(state, clear_lanes)
    print("CLEAR", clear_lanes, "TARGET", t_lane, "CURRENT", state.lane)

    # If we are in an outer lane and need to get to the other outer lane
    if state.lane != 1 and\
        (car_in_front := [c for c in state.cars if c.lane == state.lane and c.position > 0]) and\
        (car_to_side := [c for c in state.cars if abs(c.lane - state.lane) == 1 and c.position >= -s.CAR_LENGTH-50]) and\
        state.velocity.x > 0:
        car_in_front = car_in_front[0]
        car_to_side = car_to_side[0]
        # Cars might be too far away
        if car_in_front.lane not in clear_lanes and car_to_side.lane not in clear_lanes:
            print("Difficult!")
            dodge_target = s.lane_to_pos(t_lane)
            if state.lane == 0:
                dodge_target -= s.CAR_WIDTH/2 + 10
            else:
                dodge_target += s.CAR_WIDTH/2 + 10
            dy = dodge_target - state.position
            ay = 1 if dodge_target > state.position else -1
            time_to_dodge = (sqrt(8*ay*dy+state.velocity.y**2)-state.velocity.y)/(2*ay)
            # If we cannot dodge with current speed, slow down
            if abs(state.velocity.y) == 0 and time_to_dodge < (car_in_front.position - s.CAR_LENGTH-50)/state.velocity.x:
                return s.ActionType.DECELERATE
            # Where is the car to the side in the time it takes to dodge?
            side_future_pos = car_to_side.position + car_to_side.velocity * time_to_dodge
            # If its ahead of us in future, we can start turning, otherwise, stay in lane and brake
            if side_future_pos > (s.CAR_LENGTH+50):
                return s.ActionType.STEER_RIGHT if dy > 0 else s.ActionType.STEER_LEFT
            else:
                if abs(state.velocity.y) > 0:
                    return to_target_position(state, s.lane_to_pos(state.lane))
                else:
                    return s.ActionType.DECELERATE

    # If our lane is not clear, we have to change
    if t_lane != state.lane or\
        abs(state.velocity.y) > 0.001 or\
        abs(state.position - s.lane_to_pos(t_lane)) > MARGIN:
        return to_target_position(state, s.lane_to_pos(t_lane))

    # If our lane is clear, YEET
    return s.ActionType.ACCELERATE
