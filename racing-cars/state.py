from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from math import ceil, sqrt
from typing import Literal

from pelutils import DataStorage
import numpy as np
import torch


@dataclass
class Data(DataStorage):
    dt: list[float]
    position: list[float]
    velocity_x: list[float]
    velocity_y: list[float]
    car1pos_x: list[float]
    car1lane: list[float]
    car2pos_x: list[float]
    car2lane: list[float]
    car1vel: list[float]
    car2vel: list[float]
    infos: list

def to_dict(obj) -> dict:
    if hasattr(obj, "__dict__"):
        return { kw: to_dict(v) for kw, v in vars(obj).items() }
    else:
        return obj

class ActionType(str, Enum):
    ACCELERATE  = 'ACCELERATE'
    DECELERATE  = 'DECELERATE'
    STEER_RIGHT = 'STEER_RIGHT'
    STEER_LEFT  = 'STEER_LEFT'
    NOTHING     = 'NOTHING'

CAR_LENGTH = 2 * 197.834228515625
CAR_WIDTH = CAR_LENGTH * 397 / 800
SENSOR_DIST = 1000 + CAR_WIDTH / 2
LENGTH = 2000
WIDTH = 425 + 450
GRID_SIZE = 5
assert WIDTH % GRID_SIZE == 0 and LENGTH % GRID_SIZE == 0

FEATURES = 13
ALL_ACTIONS = tuple(a for a in ActionType)
ACTIONS = len(ALL_ACTIONS)

def pos_to_lane(y: float) -> Literal[0, 1, 2]:
    if y <= 280:
        return 0
    if y <= 582:
        return 1
    return 2

def lane_to_pos(lane: Literal[0, 1, 2], driver=False) -> float:
    # if driver:
    #     return { 0: 130+CAR_WIDTH/2-10, 1: 425, 2: 713-CAR_WIDTH/2+15 }[lane]
    return { 0: 130, 1: 425, 2: 733 }[lane]

@dataclass
class Feature:
    pass

@dataclass
class Vector:
    x: float
    y: float

@dataclass
class Car:
    position: float  # Center of car on x axis
    velocity: float  # In forward direction
    lane: int

@dataclass
class Sensors:
    left_side:   float
    left_front:  float | None
    front:       float | None
    right_front: float | None
    right_side:  float
    right_back:  float | None
    back:        float | None
    left_back:   float | None

    def readings(self) -> list[float | None]:
        return [self.left_side, self.left_front, self.front, self.right_front,
                self.right_side, self.right_back, self.back, self.left_back]

@dataclass
class Information:
    """ All the information given by the game """
    elapsed_time_ms: float
    velocity: Vector
    sensors: Sensors
    did_crash: bool
    distance: int

    @staticmethod
    def from_dict(d: dict) -> Information:
        return Information(
            elapsed_time_ms = d["elapsed_time_ms"],
            velocity        = Vector(**{k: float(x) for k, x in d["velocity"].items()}),
            sensors         = Sensors(**{kw: float(v) if v is not None else v for kw, v in d["sensors"].items()}),
            did_crash       = d["did_crash"],
            distance        = float(d["distance"]),
        )

@dataclass
class State:
    dt: float
    velocity: Vector
    position: float  # Center of car (positive is downwards), 0 is top wall
    cars: list[Car]
    info: Information

    def new_state(self, info: Information) -> State:
        new_state = deepcopy(self)
        dt = info.elapsed_time_ms - self.info.elapsed_time_ms
        new_state.dt = dt

        # Calculate y position
        # There will always be at least one sensor pointing on a wall
        new_state.position = self.position + info.velocity.y
        lane = pos_to_lane(new_state.position)
        # Update cars velocity
        new_state.velocity = info.velocity

        # Update information about other cars
        # Check in reverse order so they can be popped if they are >= 1000 away
        # print(range(len(new_state.cars)-1, -1, -1))
        for i in range(len(new_state.cars)-1, -1, -1):
            # First update velocity relative to our car
            car = new_state.cars[i]
            car.velocity -= new_state.velocity.x - self.velocity.x

            # Then update estimated position
            car.position += car.velocity

            # If car is too far away, pop from state
            if abs(car.position) > SENSOR_DIST:
                new_state.cars.pop(i)

        # Check for new cars
        # New cars can be detected by a small difference in any non-side facing sensor
        # Car readings are discarded if facing a wall or if a car is already registered within 50 of the position
        # breakpoint()
        for i, (old_reading, new_reading) in enumerate(zip(self.info.sensors.readings(), info.sensors.readings())):
            # print(i, old_reading, new_reading)
            # if i == 7 and new_reading < 600:
            #     breakpoint()
            # Check that not side sensor and numerical readings
            if i not in {0, 4} and isinstance(old_reading, (float, int)) and isinstance(new_reading, (float, int)):
                # Check for small change in reading
                if 0.0005 < abs(new_reading - old_reading) < 50:
                    # Check if seeing wall
                    wall_dist = 0
                    if i in {1, 7}:
                        wall_dist = new_state.position * sqrt(2)
                    elif i in {3, 5}:
                        wall_dist = (WIDTH - new_state.position) * sqrt(2)
                    if wall_dist != 0:
                        is_facing_wall = abs(wall_dist-new_reading) < 20
                    else:
                        # Front or back sensor will never see walls
                        is_facing_wall = False

                    if not is_facing_wall:
                        # We have a car, so we calculate it's position
                        # If close to an existing car's position, remove
                        # Else add to car list

                        # First calculate y position of reading. If a car exists in that lane, we discard it
                        if i in {1, 7}:
                            y_reading = new_state.position - sqrt(0.5) * new_reading
                        elif i in {3, 5}:
                            y_reading = new_state.position + sqrt(0.5) * new_reading
                        else:
                            y_reading = new_state.position
                        carlane = pos_to_lane(y_reading)
                        # It may be a new car, if we get to here
                        # In that case, we pop the existing car
                        old_car = None
                        for j, car in enumerate(new_state.cars):
                            if carlane == car.lane:
                                old_car = new_state.cars.pop(j)
                                break
                        # Calculate its x position and velocity
                        if i in {1, 3}:
                            x_reading = new_reading * sqrt(0.5)
                            old_x_reading = old_reading * sqrt(0.5)
                        elif i in {5, 7}:
                            x_reading = -new_reading * sqrt(0.5)
                            old_x_reading = -old_reading * sqrt(0.5)
                        elif i == 2:
                            x_reading = new_reading
                            old_x_reading = old_reading
                        elif i == 6:
                            x_reading = -new_reading
                            old_x_reading = -old_reading
                        velocity = x_reading - old_x_reading
                        if i in {1, 2, 3}:
                            x_reading += CAR_LENGTH / 2
                        else:
                            x_reading -= CAR_LENGTH / 2

                        new_car = Car(x_reading, velocity, carlane)
                        if old_car:
                            if abs(old_car.position - new_car.position) > 150:
                                # Definitely a new car
                                new_state.cars.append(new_car)
                            else:
                                # The same car, so use the old car to prevent risking wrong speed readings
                                new_state.cars.append(deepcopy(old_car))
                        else:
                            new_state.cars.append(new_car)

        new_state.info = info
        return new_state

    @staticmethod
    def _car_bounding_box(car_pos: Vector) -> tuple[slice, slice]:
        """ Returns a bounding box for where a car exists in the grid representation
        Coordinate system has origin in the top center
        car_pos is at center of car """
        # First, change to coordinate system that has origin in upper left corner
        car_pos = deepcopy(car_pos)
        car_pos.x += LENGTH / 2
        # Then calculate bounding boxes
        left_x   = car_pos.x - CAR_LENGTH / 2
        right_x  = car_pos.x + CAR_LENGTH / 2
        top_y    = car_pos.y - CAR_WIDTH / 2
        bottom_y = car_pos.y + CAR_WIDTH / 2
        x = slice(int(left_x // GRID_SIZE), ceil(right_x / GRID_SIZE))
        y = slice(int(top_y // GRID_SIZE), ceil(bottom_y / GRID_SIZE))
        return y, x

    def grid_representation(self) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """ Returns a grid representation of the state
        Assumes no mmore than three cars at a time
        The first element is the one-hot grid representation, which is (WIDTH x LENGTH) / GRID_SIZE x 3
        The second are the velocities of the cars, which is a three long vector
        If cars are not present, 0 is returned for velocity
        Grids have value 1 if partially or wholly covered """
        grid = torch.zeros((WIDTH // GRID_SIZE, LENGTH // GRID_SIZE, 3), dtype=torch.float32)
        # First insert our car
        # breakpoint()
        grid[(*self._car_bounding_box(Vector(0, self.position)), 0)] = 1
        # Then insert other cars
        for i, car in enumerate(self.cars, start=1):
            try:
                grid[(*self._car_bounding_box(car.position), i)] = 1
            except IndexError:
                breakpoint()

        velocities = torch.FloatTensor([self.velocity.x, self.velocity.y, *(car.velocity for car in self.cars), *(0 for _ in range(2-len(self.cars)))])

        return grid, velocities

    @property
    def lane(self):
        return pos_to_lane(self.position)
