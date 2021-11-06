import state
import train

MAX_SPEED = 80
LANE_TARGETS = [130+50, 425, 733-50]
DIFF_MARGIN = 10

def target_lane(s: state.State) -> int:
    if not s.cars:
        return 1
    lanes = list(range(3))
    for c in s.cars:
        lanes.remove(state.pos_to_lane(c.position.y))
    if len(lanes) == 1:
        return lanes[0]
    else:
        if not lanes or 1 in lanes:
            return 1
        else:
            if (l := state.pos_to_lane(s.position)) in lanes:
                return l
            else:
                return 0

def keep_speeding(distance: float, velocity: float) -> bool:
    if distance > velocity:
        return True
    return False

def act(s: state.State) -> train.ActionType:
    import time
    time.sleep(5)
    target = target_lane(s)
    diff = LANE_TARGETS[target] - s.position

    if abs(diff) > DIFF_MARGIN:
        keep_going = train.ActionType.STEER_RIGHT if s.velocity.y > 0  else train.ActionType.STEER_LEFT
        change_d = train.ActionType.STEER_RIGHT if keep_going == train.ActionType.STEER_LEFT else train.ActionType.STEER_LEFT
        # If we are driving away from target, we obviously need to fix it
        if diff*s.velocity.y < 0:
            return change_d
        else:
            if keep_speeding(abs(diff), abs(s.velocity.y)):
                return keep_going
            else:
                return change_d

    if not s.cars or all(c.position.x < 0 for c in s.cars):
        return train.ActionType.ACCELERATE
    return train.ActionType.NOTHING
