import numpy as np

try:
    from .traffic_config import (
        ARRIVAL_HIGH_EXCLUSIVE,
        ARRIVAL_LOW_INCLUSIVE,
        DECISION_INTERVAL_SECONDS,
        INITIAL_QUEUE_HIGH_EXCLUSIVE,
        INITIAL_QUEUE_LOW_INCLUSIVE,
        MAX_EPISODE_SECONDS,
        MIN_GREEN_SECONDS,
        PHASE_DURATION_CHOICES_SECONDS,
        PHASES,
        REWARD_WEIGHTS,
        STATE_NORMALIZATION,
        TARGET_MAX_GREEN_SECONDS,
        TURN_ORDER,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from traffic_config import (
        ARRIVAL_HIGH_EXCLUSIVE,
        ARRIVAL_LOW_INCLUSIVE,
        DECISION_INTERVAL_SECONDS,
        INITIAL_QUEUE_HIGH_EXCLUSIVE,
        INITIAL_QUEUE_LOW_INCLUSIVE,
        MAX_EPISODE_SECONDS,
        MIN_GREEN_SECONDS,
        PHASE_DURATION_CHOICES_SECONDS,
        PHASES,
        REWARD_WEIGHTS,
        STATE_NORMALIZATION,
        TARGET_MAX_GREEN_SECONDS,
        TURN_ORDER,
    )


class TrafficEnv:
    def __init__(self, max_steps=100):
        self.num_directions = 4  # a, b, c, d
        self.turns = TURN_ORDER
        self.phase_settings = PHASES
        self.phases = [phase["name"] for phase in self.phase_settings]
        self.duration_choices = PHASE_DURATION_CHOICES_SECONDS
        self.actions = [
            {
                "phase_index": phase_index,
                "duration_seconds": duration_seconds,
                "label": f"{phase['name']}__{duration_seconds}s",
            }
            for phase_index, phase in enumerate(self.phase_settings)
            for duration_seconds in self.duration_choices
        ]
        self.decision_interval_seconds = DECISION_INTERVAL_SECONDS
        self.max_steps = max_steps
        self.last_step_metrics = {}
        self.reset()

    def reset(self):
        # cars waiting by approach and lane movement: left, straight, right
        self.lane_counts = np.random.randint(
            INITIAL_QUEUE_LOW_INCLUSIVE,
            INITIAL_QUEUE_HIGH_EXCLUSIVE,
            size=(self.num_directions, len(self.turns)),
        )
        
        # wait time by approach and lane movement, measured in seconds
        self.lane_wait_times = np.zeros((self.num_directions, len(self.turns)))

        self.current_phase = 0
        self.time_in_phase = 0
        self.last_duration_seconds = self.decision_interval_seconds
        self.elapsed_seconds = 0
        self.steps_taken = 0
        self.last_step_metrics = {
            "cars_passed": 0,
            "overtime_seconds": 0,
            "switched_phase": False,
            "reward_components": {},
        }

        return self._get_state()

    def _get_state(self):
        # normalize for stability
        state = np.concatenate([
            self.lane_counts.flatten() / STATE_NORMALIZATION["queue"],
            self.lane_wait_times.flatten() / STATE_NORMALIZATION["wait"],
            [self.current_phase / len(self.phases)],
            [self.time_in_phase / STATE_NORMALIZATION["phase_time"]],
            [self.last_duration_seconds / max(self.duration_choices)],
        ])
        return state

    def step(self, action):
        action_info = self._decode_action(action)
        previous_phase = self.current_phase
        duration_seconds = action_info["duration_seconds"]
        switched_phase = action_info["phase_index"] != self.current_phase

        if switched_phase:
            self.current_phase = action_info["phase_index"]

        phase_pressures = self._phase_pressures()
        selected_pressure = phase_pressures[self.current_phase]
        best_pressure = max(phase_pressures)
        cars_passed = self._simulate_flow(duration_seconds)
        self._update_wait_times(duration_seconds)

        # Each action is a complete green interval. A 90s action never becomes
        # 180s just because the next decision chooses the same phase again.
        next_time_in_phase = duration_seconds
        reward, reward_components = self._calculate_reward(
            cars_passed=cars_passed,
            switched_phase=switched_phase,
            next_time_in_phase=next_time_in_phase,
            selected_pressure=selected_pressure,
            best_pressure=best_pressure,
        )

        # random new cars arriving
        arrivals = np.random.randint(
            ARRIVAL_LOW_INCLUSIVE,
            ARRIVAL_HIGH_EXCLUSIVE,
            size=(self.num_directions, len(self.turns)),
        )
        self.lane_counts += arrivals

        self.time_in_phase = duration_seconds
        self.last_duration_seconds = duration_seconds
        self.elapsed_seconds += duration_seconds
        self.steps_taken += 1
        self.last_step_metrics = {
            "cars_passed": int(cars_passed),
            "arrivals": arrivals.copy(),
            "overtime_seconds": max(0, self.time_in_phase - TARGET_MAX_GREEN_SECONDS),
            "selected_duration_seconds": duration_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "previous_phase": previous_phase,
            "current_phase": self.current_phase,
            "switched_phase": switched_phase,
            "reward_components": reward_components,
        }

        next_state = self._get_state()
        done = self.steps_taken >= self.max_steps or self.elapsed_seconds >= MAX_EPISODE_SECONDS

        return next_state, reward, done

    @property
    def car_counts(self):
        return self.lane_counts.sum(axis=1)

    @property
    def wait_times(self):
        return self.lane_wait_times.max(axis=1)

    def _decode_action(self, action):
        action_index = int(action)
        if action_index < 0 or action_index >= len(self.actions):
            raise ValueError(f"Action {action} is outside valid range 0..{len(self.actions) - 1}")
        return self.actions[action_index]

    def _simulate_flow(self, duration_seconds):
        cars_passed = 0
        phase = self.phase_settings[self.current_phase]
        effective_green_seconds = duration_seconds

        for approach_index in phase["approaches"]:
            for turn, cars_per_interval in phase["movement_capacities"].items():
                turn_index = self.turns.index(turn)
                capacity = int(
                    cars_per_interval
                    * effective_green_seconds
                    / self.decision_interval_seconds
                )
                capacity = max(1, capacity) if effective_green_seconds > 0 else 0
                passed = min(capacity, self.lane_counts[approach_index, turn_index])
                self.lane_counts[approach_index, turn_index] -= passed
                cars_passed += passed

        return cars_passed

    def _update_wait_times(self, duration_seconds):
        phase = self.phase_settings[self.current_phase]
        active_lanes = {
            (approach_index, self.turns.index(turn))
            for approach_index in phase["approaches"]
            for turn in phase["movement_capacities"]
        }
        for approach_index in range(self.num_directions):
            for turn_index in range(len(self.turns)):
                if self.lane_counts[approach_index, turn_index] <= 0:
                    self.lane_wait_times[approach_index, turn_index] = 0
                elif (approach_index, turn_index) in active_lanes:
                    self.lane_wait_times[approach_index, turn_index] = max(
                        0,
                        self.lane_wait_times[approach_index, turn_index] - duration_seconds,
                    )
                else:
                    self.lane_wait_times[approach_index, turn_index] += duration_seconds

    def _phase_pressures(self):
        pressures = []
        for phase in self.phase_settings:
            pressure = 0.0
            for approach_index in phase["approaches"]:
                for turn in phase["movement_capacities"]:
                    turn_index = self.turns.index(turn)
                    queue = self.lane_counts[approach_index, turn_index]
                    wait = self.lane_wait_times[approach_index, turn_index]
                    pressure += queue * (1.0 + wait / 60.0)
            pressures.append(float(pressure))
        return pressures

    def _calculate_reward(
        self,
        cars_passed,
        switched_phase,
        next_time_in_phase,
        selected_pressure,
        best_pressure,
    ):
        overtime_seconds = max(0, next_time_in_phase - TARGET_MAX_GREEN_SECONDS)
        early_switch_seconds = max(0, MIN_GREEN_SECONDS - next_time_in_phase)
        starvation_seconds = float(np.max(self.lane_wait_times)) if self.lane_wait_times.size else 0.0

        components = {
            "cars_passed": REWARD_WEIGHTS["cars_passed"] * cars_passed,
            "total_wait": REWARD_WEIGHTS["total_wait"] * float(np.sum(self.lane_wait_times)),
            "total_queue": REWARD_WEIGHTS["total_queue"] * float(np.sum(self.lane_counts)),
            "overtime": REWARD_WEIGHTS["overtime"] * overtime_seconds,
            "switch_before_min": REWARD_WEIGHTS["switch_before_min"] * early_switch_seconds,
            "starvation": REWARD_WEIGHTS["starvation"] * starvation_seconds,
            "served_pressure": REWARD_WEIGHTS["served_pressure"] * selected_pressure,
            "pressure_gap": REWARD_WEIGHTS["pressure_gap"] * max(0.0, best_pressure - selected_pressure),
        }
        return float(sum(components.values())), components
