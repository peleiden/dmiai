from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Literal

def to_dict(obj) -> dict:
    if hasattr(obj, "__dict__"):
        return { kw: to_dict(v) for kw, v in vars(obj).items() }
    else:
        return obj

class ActionType(str, Enum):
    ACCELERATE = 'ACCELERATE'
    DECELERATE = 'DECELERATE'
    STEER_RIGHT = 'STEER_RIGHT'
    STEER_LEFT = 'STEER_LEFT'
    NOTHING = 'NOTHING'

FEATURES = 8 * 2  # Current and previous state, each of which have eight features
ALL_ACTIONS = tuple(a for a in ActionType)

WIDTH = 425 + 450
CAR_WIDTH = 425 - 200.75
SENSOR_DIST = 1020  # Should be 1000, but add a little just in case

def pos_to_lane(y: float) -> Literal[0, 1, 2]:
    if y <= 280:
        return 0
    if y <= 582:
        return 1
    return 2

def lane_to_pos(lane: Literal[0, 1, 2]) -> float:
    return { 0: 130, 1: 450, 2: 733 }[lane]

@dataclass
class Vector:
    x: float
    y: float

@dataclass
class Car:
    velocity: float   # In forward direction
    position: Vector  # Center of car. (0, 0) is where our car stars

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
    def from_dict(d: dict) -> State:
        return Information(
            elapsed_time_ms = d["elapsed_time_ms"],
            velocity        = Vector(**{k: float(x) for k, x in d["velocity"].items()}),
            sensors         = Sensors(**{kw: float(v) if v is not None else v for kw, v in d["sensors"].items()}),
            did_crash       = d["did_crash"],
            distance        = float(d["distance"]),
        )

@dataclass
class State:
    velocity: Vector
    position: float  # Center of car (positive is downwards), 0 is top wall
    cars: list[Car]
    info: Information

    def new_state(self, info: Information) -> State:
        new_state = deepcopy(self)
        dt = info.elapsed_time_ms - self.info.elapsed_time_ms

        # Calculate expected y position
        # If close to what sensors say, use their values, else use expected position
        new_state.position += info.velocity.y * dt
        if abs(new_state.position - info.sensors.left_side) < 10:
            new_state.position = info.sensors.left_side
        elif abs(new_state.position - (WIDTH-info.sensors.right_side)) < 10:
            new_state.position = WIDTH - info.sensors.right_side

        # Update cars velocity
        new_state.velocity = info.velocity

        # Update information about other cars
        # Check in reverse order so they can be popped if they are >= 1000 away
        for i in range(len(new_state.cars)-1, -1, -1):
            # First update velocity relative to our car
            car = self.cars[i]
            car.velocity += self.velocity.x - new_state.velocity.x

            # Then update estimated position
            car.position.x += car.velocity * dt

            # If car is too far away, pop from state
            if abs(car.position.x) > SENSOR_DIST:
                new_state.cars.pop(i)

        # Check for new cars
        # New cars can be detected by a small difference in any non-side facing sensor
        # Car readings are discarded if facing a wall or if a car is already registered within 50 of the position
        for i, (old_reading, new_reading) in enumerate(zip(self.info.sensors.readings(), info.sensors.readings())):
            # Check that not side sensor and numerical readings
            if i not in {0, 4} and isinstance(old_reading, (float, int)) and isinstance(new_reading, (float, int)):
                # Check for small change in reading
                if 0 < abs(new_reading - old_reading) < 50:
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
                        if not any(pos_to_lane(car.position.y) == carlane for car in self.cars):
                            # It is a new car, if we get to here
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
                            velocity = (x_reading-old_x_reading) / dt
                            new_state.cars.append(Car(velocity, Vector(x_reading, lane_to_pos(carlane))))

        new_state.info = info
        return new_state



