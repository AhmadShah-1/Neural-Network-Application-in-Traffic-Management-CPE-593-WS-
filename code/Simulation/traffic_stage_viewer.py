"""Step-by-step traffic simulation viewer for the DQN controller."""

from __future__ import annotations

import copy
import math
import os
import random
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

import pygame
import torch

try:
    from State.traffic_config import DECISION_INTERVAL_SECONDS, PHASE_DURATION_CHOICES_SECONDS, PHASES, STATE_NORMALIZATION
except ImportError:  # pragma: no cover - supports running from inside code/Simulation
    from ..State.traffic_config import DECISION_INTERVAL_SECONDS, PHASE_DURATION_CHOICES_SECONDS, PHASES, STATE_NORMALIZATION


WIDTH, HEIGHT = 1240, 860
PANEL_X = 900
ROAD_RIGHT = PANEL_X - 8
CENTER = np.array([450.0, 420.0])
ROAD = 190
LANE = 30
STOP_GAP = 82
QUEUE_SPACING = 31
CAR_W, CAR_H = 18, 29
PHASE_SECONDS = DECISION_INTERVAL_SECONDS
MANUAL_STEP_SECONDS = 1.0
VISUAL_STEP_POINTS = 7

APPROACH_ORDER = ["A", "B", "C", "D"]
TURN_ORDER = ["left", "straight", "right"]
TURN_LABELS = {"left": "L", "straight": "S", "right": "R"}
TURN_NAMES = {"left": "Left", "straight": "Straight", "right": "Right"}

COLORS = {
    "grass": (213, 226, 199),
    "road": (47, 50, 53),
    "road_edge": (32, 35, 37),
    "lane": (237, 220, 142),
    "stop": (248, 248, 242),
    "panel": (250, 247, 235),
    "panel_line": (203, 196, 178),
    "ink": (28, 31, 34),
    "muted": (98, 105, 111),
    "green": (48, 179, 101),
    "red": (210, 66, 66),
    "amber": (234, 170, 50),
    "blue": (50, 122, 218),
    "purple": (135, 84, 194),
    "orange": (229, 129, 48),
    "white": (255, 255, 255),
    "disabled": (172, 170, 161),
}

TURN_COLORS = {
    "left": COLORS["purple"],
    "straight": COLORS["blue"],
    "right": COLORS["orange"],
}

APPROACH_COLORS = {
    "A": COLORS["orange"],
    "B": COLORS["purple"],
    "C": COLORS["green"],
    "D": COLORS["blue"],
}

APPROACHES = {
    "A": {
        "name": "A / North",
        "stop": np.array([CENTER[0] - LANE * 1.5, CENTER[1] - STOP_GAP]),
        "heading": np.array([0.0, 1.0]),
        "count_pos": (545, 92),
    },
    "B": {
        "name": "B / West",
        "stop": np.array([CENTER[0] - STOP_GAP, CENTER[1] + LANE * 1.5]),
        "heading": np.array([1.0, 0.0]),
        "count_pos": (82, 275),
    },
    "C": {
        "name": "C / South",
        "stop": np.array([CENTER[0] + LANE * 1.5, CENTER[1] + STOP_GAP]),
        "heading": np.array([0.0, -1.0]),
        "count_pos": (542, 742),
    },
    "D": {
        "name": "D / East",
        "stop": np.array([CENTER[0] + STOP_GAP, CENTER[1] - LANE * 1.5]),
        "heading": np.array([-1.0, 0.0]),
        "count_pos": (703, 540),
    },
}

INITIAL_LANES = {
    "A": {"left": 3, "straight": 7, "right": 2},
    "B": {"left": 2, "straight": 4, "right": 1},
    "C": {"left": 4, "straight": 6, "right": 3},
    "D": {"left": 1, "straight": 5, "right": 2},
}

INITIAL_WAITS = np.array([14.0, 6.0, 18.0, 9.0], dtype=float)
INITIAL_LANE_WAITS = np.repeat(INITIAL_WAITS[:, None], len(TURN_ORDER), axis=1)

ARRIVAL_PATTERNS = [
    {"A": ["straight"], "B": ["left"], "C": ["right"], "D": ["straight", "right"]},
    {"A": ["left", "right"], "B": ["straight"], "C": ["straight"], "D": ["left"]},
    {"A": ["straight"], "B": ["straight", "right"], "C": ["left"], "D": ["straight"]},
    {"A": ["right"], "B": ["left"], "C": ["straight", "right"], "D": ["straight"]},
    {"A": ["left"], "B": ["straight"], "C": ["left", "straight"], "D": ["right"]},
]

DESTINATIONS = {
    ("A", "straight"): "C",
    ("A", "left"): "D",
    ("A", "right"): "B",
    ("B", "straight"): "D",
    ("B", "left"): "A",
    ("B", "right"): "C",
    ("C", "straight"): "A",
    ("C", "left"): "B",
    ("C", "right"): "D",
    ("D", "straight"): "B",
    ("D", "left"): "C",
    ("D", "right"): "A",
}


@dataclass(frozen=True)
class Vehicle:
    id: int
    approach: str
    turn: str


@dataclass(frozen=True)
class Movement:
    vehicle: Vehicle
    destination: str
    order: int


@dataclass(frozen=True)
class ActiveMovement:
    vehicle: Vehicle
    destination: str
    order: int
    path_index: int


@dataclass
class StageSnapshot:
    stage: int
    queues: dict[str, dict[str, list[Vehicle]]]
    wait_times: np.ndarray
    current_phase: int
    time_in_phase: float
    state_vector: np.ndarray
    q_values: np.ndarray
    recommended_action: int
    recommended_action_index: int
    recommended_duration_seconds: int
    applied_action: int | None
    served_movements: list[Movement]
    active_movements: list[ActiveMovement]
    arrivals: list[Vehicle]
    passed_counts: dict[str, dict[str, int]]
    note: str


@dataclass(frozen=True)
class Button:
    rect: pygame.Rect
    label: str


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def perp_left(heading):
    hx, hy = heading
    return np.array([hy, -hx], dtype=float)


def lane_position(approach, turn, queue_index=0):
    info = APPROACHES[approach]
    heading = unit(info["heading"])
    left = perp_left(heading)
    # Right-hand traffic assumption: each approach has three inbound lanes.
    # Left-turn cars sit on the driver's left; right-turn cars sit on the driver's right.
    offsets = {"left": LANE, "straight": 0, "right": -LANE}
    return info["stop"] - heading * (queue_index * QUEUE_SPACING) + left * offsets[turn]


def exit_point(approach, turn):
    southbound_x = lane_position("A", "straight", 0)[0]
    northbound_x = lane_position("C", "straight", 0)[0]
    eastbound_y = lane_position("B", "straight", 0)[1]
    westbound_y = lane_position("D", "straight", 0)[1]
    exits = {
        ("A", "straight"): np.array([southbound_x, HEIGHT + 55.0]),
        ("A", "left"): np.array([ROAD_RIGHT + 45.0, eastbound_y]),
        ("A", "right"): np.array([-55.0, westbound_y]),
        ("B", "straight"): np.array([ROAD_RIGHT + 45.0, eastbound_y]),
        ("B", "left"): np.array([northbound_x, -55.0]),
        ("B", "right"): np.array([southbound_x, HEIGHT + 55.0]),
        ("C", "straight"): np.array([northbound_x, -55.0]),
        ("C", "left"): np.array([-55.0, westbound_y]),
        ("C", "right"): np.array([ROAD_RIGHT + 45.0, eastbound_y]),
        ("D", "straight"): np.array([-55.0, westbound_y]),
        ("D", "left"): np.array([southbound_x, HEIGHT + 55.0]),
        ("D", "right"): np.array([northbound_x, -55.0]),
    }
    return exits[(approach, turn)]


def bezier(points, samples=58):
    points = [np.asarray(point, dtype=float) for point in points]
    if len(points) == 2:
        return [points[0] + (points[1] - points[0]) * t for t in np.linspace(0, 1, samples)]
    p0, p1, p2 = points
    return [(1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2 for t in np.linspace(0, 1, samples)]


def travel_path(approach, turn):
    start = lane_position(approach, turn, 0)
    end = exit_point(approach, turn)
    if turn == "straight":
        return bezier([start, end], samples=64)
    heading = unit(APPROACHES[approach]["heading"])
    control = CENTER + heading * 30 if turn == "left" else CENTER - heading * 20
    return bezier([start, control, end], samples=64)


def phase_allows(phase, approach, turn):
    if phase < 0 or phase >= len(PHASES):
        return False
    approach_index = APPROACH_ORDER.index(approach)
    phase_config = PHASES[phase]
    return approach_index in phase_config["approaches"] and turn in phase_config["movement_capacities"]


def phase_name(phase_names, phase):
    return phase_names[phase] if 0 <= phase < len(phase_names) else f"phase_{phase}"


class StageTrafficSimulation:
    """Deterministic, reversible simulation driven by DQN decisions."""

    def __init__(self, decision_fn: Callable[[np.ndarray], tuple[int, np.ndarray]], phase_names, seed=19):
        self.decision_fn = decision_fn
        self.phase_names = list(phase_names)
        self.seed = seed
        self.rng = random.Random(seed)
        self.next_vehicle_id = 1
        self.snapshots: list[StageSnapshot] = []
        self.current_index = 0
        self.reset()

    @property
    def current(self):
        return self.snapshots[self.current_index]

    def reset(self):
        self.rng = random.Random(self.seed)
        self.next_vehicle_id = 1
        queues = self._initial_queues()
        self.snapshots = [
            self._make_snapshot(
                stage=0,
                queues=queues,
                wait_times=INITIAL_LANE_WAITS.copy(),
                current_phase=0,
                time_in_phase=0.0,
                applied_action=None,
                served_movements=[],
                active_movements=[],
                arrivals=[],
                passed_counts=self._empty_passed_counts(),
                note="Initial queued traffic on all four approaches.",
            )
        ]
        self.current_index = 0

    def can_go_back(self):
        return self.current_index > 0

    def can_go_forward(self):
        return True

    def previous(self):
        if self.can_go_back():
            self.current_index -= 1

    def advance(self):
        if self.current_index < len(self.snapshots) - 1:
            self.current_index += 1
            return

        base = self.current
        queues = copy.deepcopy(base.queues)
        wait_times = base.wait_times.copy()
        passed_counts = copy.deepcopy(base.passed_counts)
        current_phase = base.current_phase
        time_in_phase = base.time_in_phase
        applied_action = None
        served_movements: list[Movement] = []
        arrivals: list[Vehicle] = []
        note = ""

        if base.active_movements:
            active_movements = self._advance_active_movements(base.active_movements)
            time_in_phase += MANUAL_STEP_SECONDS
            note = "Moved active cars one manual step. Queues stay fixed until the next release."
        else:
            action = base.recommended_action
            duration_seconds = base.recommended_duration_seconds
            current_phase = action
            time_in_phase = duration_seconds
            applied_action = action
            served_movements = self._release_cars(queues, action, duration_seconds)
            active_movements = [
                ActiveMovement(movement.vehicle, movement.destination, movement.order, path_index=-(movement.order * 4))
                for movement in served_movements
            ]
            arrivals = self._arrivals_for_stage(base.stage + 1)
            for vehicle in arrivals:
                queues[vehicle.approach][vehicle.turn].append(vehicle)
            for movement in served_movements:
                passed_counts[movement.vehicle.approach][movement.vehicle.turn] += 1
            wait_times = self._wait_times_after_release(queues, wait_times, action, duration_seconds)
            note = f"Released cars for {phase_name(self.phase_names, action)} for {duration_seconds}s. Use Step Forward to move them."

        snapshot = self._make_snapshot(
            stage=base.stage + 1,
            queues=queues,
            wait_times=wait_times,
            current_phase=current_phase,
            time_in_phase=time_in_phase,
            applied_action=applied_action,
            served_movements=served_movements,
            active_movements=active_movements,
            arrivals=arrivals,
            passed_counts=passed_counts,
            note=note,
        )
        self.snapshots.append(snapshot)
        self.current_index += 1

    def lane_counts(self, snapshot=None):
        snapshot = snapshot or self.current
        return {
            approach: {turn: len(snapshot.queues[approach][turn]) for turn in TURN_ORDER}
            for approach in APPROACH_ORDER
        }

    def queue_totals(self, snapshot=None):
        counts = self.lane_counts(snapshot)
        return np.array([sum(counts[approach].values()) for approach in APPROACH_ORDER], dtype=float)

    def _initial_queues(self):
        queues = {approach: {turn: [] for turn in TURN_ORDER} for approach in APPROACH_ORDER}
        for approach in APPROACH_ORDER:
            for turn in TURN_ORDER:
                for _ in range(INITIAL_LANES[approach][turn]):
                    queues[approach][turn].append(self._new_vehicle(approach, turn))
        return queues

    def _new_vehicle(self, approach, turn):
        vehicle = Vehicle(self.next_vehicle_id, approach, turn)
        self.next_vehicle_id += 1
        return vehicle

    def _empty_passed_counts(self):
        return {approach: {turn: 0 for turn in TURN_ORDER} for approach in APPROACH_ORDER}

    def _state_from(self, queues, wait_times, current_phase, time_in_phase):
        lane_counts = np.array(
            [len(queues[approach][turn]) for approach in APPROACH_ORDER for turn in TURN_ORDER],
            dtype=float,
        )
        lane_waits = np.asarray(wait_times, dtype=float).reshape(len(APPROACH_ORDER), len(TURN_ORDER)).flatten()
        return np.concatenate(
            [
                lane_counts / STATE_NORMALIZATION["queue"],
                lane_waits / STATE_NORMALIZATION["wait"],
                [current_phase / len(self.phase_names)],
                [time_in_phase / STATE_NORMALIZATION["phase_time"]],
                [max(PHASE_SECONDS, time_in_phase) / max(PHASE_DURATION_CHOICES_SECONDS)],
            ]
        )

    def _make_snapshot(
        self,
        stage,
        queues,
        wait_times,
        current_phase,
        time_in_phase,
        applied_action,
        served_movements,
        active_movements,
        arrivals,
        passed_counts,
        note,
    ):
        state_vector = self._state_from(queues, wait_times, current_phase, time_in_phase)
        recommended_action_index, q_values = self.decision_fn(state_vector)
        recommended_action, recommended_duration_seconds = self._decode_model_action(
            recommended_action_index,
            len(q_values),
        )
        return StageSnapshot(
            stage=stage,
            queues=copy.deepcopy(queues),
            wait_times=wait_times.copy(),
            current_phase=current_phase,
            time_in_phase=float(time_in_phase),
            state_vector=state_vector,
            q_values=np.asarray(q_values, dtype=float),
            recommended_action=int(recommended_action),
            recommended_action_index=int(recommended_action_index),
            recommended_duration_seconds=int(recommended_duration_seconds),
            applied_action=applied_action,
            served_movements=list(served_movements),
            active_movements=list(active_movements),
            arrivals=list(arrivals),
            passed_counts=copy.deepcopy(passed_counts),
            note=note,
        )

    def _advance_active_movements(self, active_movements):
        advanced = []
        for movement in active_movements:
            path_len = len(travel_path(movement.vehicle.approach, movement.vehicle.turn))
            next_index = movement.path_index + VISUAL_STEP_POINTS
            if next_index < path_len - 1:
                advanced.append(
                    ActiveMovement(
                        movement.vehicle,
                        movement.destination,
                        movement.order,
                        path_index=next_index,
                    )
                )
        return advanced

    def _wait_times_after_release(self, queues, wait_times, action, duration_seconds=PHASE_SECONDS):
        updated = wait_times.copy()
        phase = PHASES[action]
        active_lanes = {
            (approach_index, TURN_ORDER.index(turn))
            for approach_index in phase["approaches"]
            for turn in phase["movement_capacities"]
        }
        for approach_index, approach in enumerate(APPROACH_ORDER):
            for turn_index, turn in enumerate(TURN_ORDER):
                if not queues[approach][turn]:
                    updated[approach_index, turn_index] = 0
                elif (approach_index, turn_index) in active_lanes:
                    updated[approach_index, turn_index] = max(0, updated[approach_index, turn_index] - duration_seconds)
                else:
                    updated[approach_index, turn_index] += duration_seconds
        return updated

    def _decode_model_action(self, action_index, q_value_count):
        if q_value_count == len(self.phase_names):
            return int(action_index), PHASE_SECONDS
        duration_count = len(PHASE_DURATION_CHOICES_SECONDS)
        return (
            int(action_index) // duration_count,
            PHASE_DURATION_CHOICES_SECONDS[int(action_index) % duration_count],
        )

    def _release_cars(self, queues, action, duration_seconds=None):
        duration_seconds = duration_seconds or PHASE_SECONDS
        movements: list[Movement] = []
        if action < 0 or action >= len(PHASES):
            return movements

        phase = PHASES[action]
        duration_multiplier = max(1, int(duration_seconds / PHASE_SECONDS))
        for approach_index in phase["approaches"]:
            approach = APPROACH_ORDER[approach_index]
            for turn, cars_per_interval in phase["movement_capacities"].items():
                capacity = cars_per_interval * duration_multiplier
                movements.extend(self._release_from_turn(queues, approach, turn, capacity))

        ordered = []
        for order, movement in enumerate(movements):
            ordered.append(Movement(movement.vehicle, movement.destination, order))
        return ordered

    def _release_from_turn(self, queues, approach, turn, capacity):
        released: list[Movement] = []
        while len(released) < capacity:
            if not queues[approach][turn]:
                break
            vehicle = queues[approach][turn].pop(0)
            released.append(Movement(vehicle, DESTINATIONS[(approach, turn)], len(released)))
        return released

    def _arrivals_for_stage(self, stage):
        pattern = ARRIVAL_PATTERNS[(stage - 1) % len(ARRIVAL_PATTERNS)]
        arrivals = []
        for approach in APPROACH_ORDER:
            for turn in pattern.get(approach, []):
                arrivals.append(self._new_vehicle(approach, turn))
        return arrivals


def make_model_decision(model):
    def decide(state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor).squeeze(0).detach().cpu().numpy()
        action = int(np.argmax(q_values))
        return action, q_values

    return decide


def draw_rotated_car(screen, pos, angle, color, label=None, font=None):
    surf = pygame.Surface((CAR_W, CAR_H), pygame.SRCALPHA)
    pygame.draw.rect(surf, color, (0, 0, CAR_W, CAR_H), border_radius=5)
    pygame.draw.rect(surf, (234, 244, 255), (3, 4, CAR_W - 6, 7), border_radius=3)
    pygame.draw.circle(surf, (25, 25, 25), (3, 6), 2)
    pygame.draw.circle(surf, (25, 25, 25), (CAR_W - 3, 6), 2)
    pygame.draw.circle(surf, (25, 25, 25), (3, CAR_H - 6), 2)
    pygame.draw.circle(surf, (25, 25, 25), (CAR_W - 3, CAR_H - 6), 2)
    rotated = pygame.transform.rotate(surf, -angle)
    rect = rotated.get_rect(center=(int(pos[0]), int(pos[1])))
    screen.blit(rotated, rect)
    if label and font:
        text = font.render(label, True, COLORS["white"])
        screen.blit(text, text.get_rect(center=rect.center))


def vehicle_angle_from_heading(heading):
    return math.degrees(math.atan2(heading[1], heading[0])) + 90


def draw_queue_vehicle(screen, vehicle, queue_index, fonts):
    pos = lane_position(vehicle.approach, vehicle.turn, queue_index)
    angle = vehicle_angle_from_heading(APPROACHES[vehicle.approach]["heading"])
    draw_rotated_car(screen, pos, angle, APPROACH_COLORS[vehicle.approach], TURN_LABELS[vehicle.turn], fonts["tiny"])


def draw_path_line(screen, path, color, end_index):
    points = [(int(p[0]), int(p[1])) for p in path[: max(2, end_index + 1)]]
    pygame.draw.lines(screen, color, False, points, 4)
    if len(points) >= 2:
        p1 = np.array(points[-2], dtype=float)
        p2 = np.array(points[-1], dtype=float)
        delta = unit(p2 - p1)
        left = perp_left(delta)
        arrow = [p2, p2 - delta * 14 + left * 7, p2 - delta * 14 - left * 7]
        pygame.draw.polygon(screen, color, [(int(p[0]), int(p[1])) for p in arrow])


def draw_movement(screen, movement, fonts):
    vehicle = movement.vehicle
    path = travel_path(vehicle.approach, vehicle.turn)
    end_index = min(len(path) - 2, max(1, movement.path_index))
    color = APPROACH_COLORS[vehicle.approach]
    draw_path_line(screen, path, color, end_index)
    pos = path[end_index]
    delta = path[min(end_index + 1, len(path) - 1)] - path[max(0, end_index - 1)]
    angle = math.degrees(math.atan2(delta[1], delta[0])) + 90
    draw_rotated_car(screen, pos, angle, color, TURN_LABELS[vehicle.turn], fonts["tiny"])


def movement_progress_percent(movement):
    path_len = len(travel_path(movement.vehicle.approach, movement.vehicle.turn))
    clamped = min(path_len - 1, max(0, movement.path_index))
    return int(round(100 * clamped / max(1, path_len - 1)))


def draw_roads(screen, fonts):
    screen.fill(COLORS["grass"])
    pygame.draw.rect(screen, COLORS["road_edge"], (CENTER[0] - ROAD / 2 - 5, 0, ROAD + 10, HEIGHT))
    pygame.draw.rect(screen, COLORS["road_edge"], (0, CENTER[1] - ROAD / 2 - 5, ROAD_RIGHT, ROAD + 10))
    pygame.draw.rect(screen, COLORS["road"], (CENTER[0] - ROAD / 2, 0, ROAD, HEIGHT))
    pygame.draw.rect(screen, COLORS["road"], (0, CENTER[1] - ROAD / 2, ROAD_RIGHT, ROAD))
    pygame.draw.rect(screen, COLORS["road"], (CENTER[0] - 92, CENTER[1] - 92, 184, 184))

    for approach in APPROACH_ORDER:
        heading = unit(APPROACHES[approach]["heading"])
        for turn in TURN_ORDER:
            start = lane_position(approach, turn, 11)
            end = lane_position(approach, turn, 0) + heading * 58
            pygame.draw.line(screen, (214, 205, 150), start.astype(int), end.astype(int), 2)

    pygame.draw.line(screen, COLORS["stop"], (CENTER[0] - 62, CENTER[1] - STOP_GAP), (CENTER[0] + 62, CENTER[1] - STOP_GAP), 5)
    pygame.draw.line(screen, COLORS["stop"], (CENTER[0] - 62, CENTER[1] + STOP_GAP), (CENTER[0] + 62, CENTER[1] + STOP_GAP), 5)
    pygame.draw.line(screen, COLORS["stop"], (CENTER[0] - STOP_GAP, CENTER[1] - 62), (CENTER[0] - STOP_GAP, CENTER[1] + 62), 5)
    pygame.draw.line(screen, COLORS["stop"], (CENTER[0] + STOP_GAP, CENTER[1] - 62), (CENTER[0] + STOP_GAP, CENTER[1] + 62), 5)

    for approach in APPROACH_ORDER:
        for turn in TURN_ORDER:
            pos = lane_position(approach, turn, 3)
            label = fonts["small"].render(TURN_LABELS[turn], True, COLORS["white"])
            screen.blit(label, label.get_rect(center=(int(pos[0]), int(pos[1]))))


def signal_positions(approach):
    heading = unit(APPROACHES[approach]["heading"])
    return {turn: lane_position(approach, turn, 0) + heading * 34 for turn in TURN_ORDER}


def draw_oriented_signal(screen, pos, heading, active, label, fonts):
    heading = unit(heading)
    left = perp_left(heading)
    half_along = 8
    half_across = 14
    points = [
        pos - left * half_across - heading * half_along,
        pos + left * half_across - heading * half_along,
        pos + left * half_across + heading * half_along,
        pos - left * half_across + heading * half_along,
    ]
    pygame.draw.polygon(screen, (25, 26, 28), [(int(p[0]), int(p[1])) for p in points])
    pygame.draw.circle(screen, COLORS["green"] if active else COLORS["red"], pos.astype(int), 7)
    text_pos = pos - heading * 18
    text = fonts["tiny"].render(label, True, COLORS["white"])
    screen.blit(text, text.get_rect(center=text_pos.astype(int)))


def draw_signals(screen, snapshot, fonts):
    for approach in APPROACH_ORDER:
        heading = APPROACHES[approach]["heading"]
        for turn, pos in signal_positions(approach).items():
            active = phase_allows(snapshot.current_phase, approach, turn)
            draw_oriented_signal(screen, pos, heading, active, TURN_LABELS[turn], fonts)
        center = signal_positions(approach)["straight"] - unit(heading) * 40
        label = fonts["small"].render(approach, True, COLORS["ink"])
        screen.blit(label, label.get_rect(center=center.astype(int)))


def draw_queue_counts(screen, snapshot, fonts):
    for approach in APPROACH_ORDER:
        lanes = {turn: len(snapshot.queues[approach][turn]) for turn in TURN_ORDER}
        total = sum(lanes.values())
        text = f"{approach}: {total}  L{lanes['left']} S{lanes['straight']} R{lanes['right']}"
        x, y = APPROACHES[approach]["count_pos"]
        surf = fonts["small"].render(text, True, COLORS["ink"])
        rect = surf.get_rect(topleft=(x, y)).inflate(14, 8)
        pygame.draw.rect(screen, COLORS["panel"], rect, border_radius=8)
        pygame.draw.rect(screen, COLORS["panel_line"], rect, 1, border_radius=8)
        screen.blit(surf, (x, y))


def draw_queues(screen, snapshot, fonts):
    max_visible = 13
    for approach in APPROACH_ORDER:
        for turn in TURN_ORDER:
            cars = snapshot.queues[approach][turn]
            for queue_index, vehicle in enumerate(cars[:max_visible]):
                draw_queue_vehicle(screen, vehicle, queue_index, fonts)
            if len(cars) > max_visible:
                pos = lane_position(approach, turn, max_visible)
                text = fonts["small"].render(f"+{len(cars) - max_visible}", True, COLORS["white"])
                pygame.draw.circle(screen, COLORS["amber"], pos.astype(int), 15)
                screen.blit(text, text.get_rect(center=pos.astype(int)))


def draw_legend(screen, fonts):
    x, y = 25, 24
    for approach in APPROACH_ORDER:
        pygame.draw.rect(screen, APPROACH_COLORS[approach], (x, y + 3, 18, 18), border_radius=4)
        label = fonts["small"].render(f"{approach} cars = {APPROACHES[approach]['name']}", True, COLORS["ink"])
        screen.blit(label, (x + 26, y))
        y += 25
    lane_note = fonts["small"].render("Lane labels: L left, S straight, R right", True, COLORS["ink"])
    screen.blit(lane_note, (x, y + 6))


def draw_button(screen, button, fonts, enabled=True):
    color = COLORS["ink"] if enabled else COLORS["disabled"]
    fill = (242, 238, 224) if enabled else (225, 222, 212)
    pygame.draw.rect(screen, fill, button.rect, border_radius=10)
    pygame.draw.rect(screen, color, button.rect, 2, border_radius=10)
    text = fonts["button"].render(button.label, True, color)
    screen.blit(text, text.get_rect(center=button.rect.center))


def panel_writer(screen, fonts, start_x=922, start_y=22):
    y = start_y

    def write(text, color=COLORS["ink"], dy=24, font_key="body"):
        nonlocal y
        surf = fonts[font_key].render(text, True, color)
        screen.blit(surf, (start_x, y))
        y += dy

    return write


def draw_panel(screen, sim, buttons, fonts):
    snapshot = sim.current
    pygame.draw.rect(screen, COLORS["panel"], (PANEL_X, 0, WIDTH - PANEL_X, HEIGHT))
    pygame.draw.line(screen, COLORS["panel_line"], (PANEL_X, 0), (PANEL_X, HEIGHT), 3)

    write = panel_writer(screen, fonts)
    write("DQN Traffic Viewer", dy=34, font_key="title")
    write(f"Step {snapshot.stage}", dy=25)
    write(f"Active: {phase_name(sim.phase_names, snapshot.current_phase)}", COLORS["green"], dy=25)
    write(f"Green time: {snapshot.time_in_phase:4.0f}s", COLORS["green"], dy=27)
    write(
        f"Next: {phase_name(sim.phase_names, snapshot.recommended_action)}"
        f" / {snapshot.recommended_duration_seconds}s",
        COLORS["muted"],
        dy=30,
    )

    write("Queues and Wait", COLORS["muted"], dy=22, font_key="small")
    counts = sim.lane_counts(snapshot)
    for i, approach in enumerate(APPROACH_ORDER):
        lanes = counts[approach]
        total = sum(lanes.values())
        max_wait = float(np.max(snapshot.wait_times[i]))
        write(
            f"{approach} L{lanes['left']:02d} S{lanes['straight']:02d} R{lanes['right']:02d}"
            f" | T{total:02d} | W{max_wait:04.1f}s",
            dy=21,
            font_key="mono",
        )

    served = len(snapshot.served_movements)
    moving = len(snapshot.active_movements)
    arrivals = len(snapshot.arrivals)
    write("", dy=5)
    write("Step Summary", COLORS["muted"], dy=22, font_key="small")
    write(f"released:{served:02d} moving:{moving:02d} arrivals:{arrivals:02d}", dy=21, font_key="mono")
    if snapshot.applied_action is not None:
        write(f"applied: {phase_name(sim.phase_names, snapshot.applied_action)}", dy=21, font_key="mono")

    if snapshot.arrivals:
        grouped = [f"{vehicle.approach}{TURN_LABELS[vehicle.turn]}" for vehicle in snapshot.arrivals]
        write("arr: " + ", ".join(grouped[:12]), dy=24, font_key="mono")

    write("", dy=5)
    write("Top Actions", COLORS["muted"], dy=22, font_key="small")
    duration_count = len(PHASE_DURATION_CHOICES_SECONDS)
    top_indices = np.argsort(snapshot.q_values)[-4:][::-1]
    for i in top_indices:
        value = snapshot.q_values[i]
        if len(snapshot.q_values) == len(sim.phase_names):
            phase_index = i
            duration = PHASE_SECONDS
        else:
            phase_index = i // duration_count
            duration = PHASE_DURATION_CHOICES_SECONDS[i % duration_count]
        marker = "*" if i == snapshot.recommended_action_index else " "
        write(f"{marker}{phase_name(sim.phase_names, phase_index):10s} {duration:02d}s {value:7.2f}", dy=20, font_key="mono")

    write("", dy=4)
    write("State", COLORS["muted"], dy=22, font_key="small")
    write(f"{len(snapshot.state_vector)} inputs: 12 queues, 12 waits, phase/time/duration", dy=38, font_key="mono")

    draw_button(screen, buttons["prev"], fonts, enabled=sim.can_go_back())
    draw_button(screen, buttons["next"], fonts, enabled=sim.can_go_forward())


def draw_scene(screen, sim, buttons, fonts):
    snapshot = sim.current
    draw_roads(screen, fonts)
    draw_legend(screen, fonts)
    draw_signals(screen, snapshot, fonts)
    for movement in snapshot.active_movements:
        draw_movement(screen, movement, fonts)
    draw_queues(screen, snapshot, fonts)
    draw_queue_counts(screen, snapshot, fonts)
    draw_panel(screen, sim, buttons, fonts)


def build_buttons():
    return {
        "prev": Button(pygame.Rect(922, 788, 138, 42), "Step Back"),
        "next": Button(pygame.Rect(1074, 788, 144, 42), "Step Forward"),
    }


def build_fonts():
    return {
        "title": pygame.font.SysFont("segoeui", 25, bold=True),
        "body": pygame.font.SysFont("segoeui", 18),
        "small": pygame.font.SysFont("segoeui", 15),
        "button": pygame.font.SysFont("segoeui", 18, bold=True),
        "mono": pygame.font.SysFont("consolas", 14),
        "tiny": pygame.font.SysFont("consolas", 9, bold=True),
    }


def run_stage_viewer(model, phase_names, seed=19):
    """Launch the manual step-by-step Pygame traffic viewer."""

    pygame.init()
    pygame.display.set_caption("DQN Manual Traffic Stepper")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    fonts = build_fonts()
    buttons = build_buttons()
    sim = StageTrafficSimulation(make_model_decision(model), phase_names, seed=seed)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_RIGHT, pygame.K_SPACE]:
                    sim.advance()
                elif event.key == pygame.K_LEFT:
                    sim.previous()
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if buttons["next"].rect.collidepoint(pos):
                    sim.advance()
                elif buttons["prev"].rect.collidepoint(pos):
                    sim.previous()

        draw_scene(screen, sim, buttons, fonts)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
