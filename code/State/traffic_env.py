import numpy as np
import random

class TrafficEnv:
    def __init__(self, max_steps=100):
        self.num_directions = 4  # a, b, c, d
        self.phases = [
            "AC_forward",
            "BD_forward",
            "AC_left",
            "BD_left"
        ]
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        # cars waiting in each direction
        self.car_counts = np.random.randint(0, 10, size=4)
        
        # wait time per direction
        self.wait_times = np.zeros(4)

        self.current_phase = 0
        self.time_in_phase = 0
        self.steps_taken = 0

        return self._get_state()

    def _get_state(self):
        # normalize for stability
        state = np.concatenate([
            self.car_counts / 20.0,
            self.wait_times / 50.0,
            [self.current_phase / len(self.phases)],
            [self.time_in_phase / 50.0]
        ])
        return state

    def step(self, action):
        reward = 0

        # change phase if action != current
        if action != self.current_phase:
            self.current_phase = action
            self.time_in_phase = 0

        # simulate traffic flow
        cars_passed = self._simulate_flow()

        # update wait times
        self.wait_times += 1

        # reward function
        reward = -np.sum(self.wait_times) - 0.5 * np.sum(self.car_counts) + 2 * cars_passed

        # random new cars arriving
        arrivals = np.random.randint(0, 3, size=4)
        self.car_counts += arrivals

        self.time_in_phase += 1
        self.steps_taken += 1

        next_state = self._get_state()
        done = self.steps_taken >= self.max_steps

        return next_state, reward, done

    def _simulate_flow(self):
        cars_passed = 0

        if self.current_phase == 0:  # AC forward
            for i in [0, 2]:
                passed = min(2, self.car_counts[i])
                self.car_counts[i] -= passed
                self.wait_times[i] = 0
                cars_passed += passed

        elif self.current_phase == 1:  # BD forward
            for i in [1, 3]:
                passed = min(2, self.car_counts[i])
                self.car_counts[i] -= passed
                self.wait_times[i] = 0
                cars_passed += passed

        elif self.current_phase == 2:  # AC left
            for i in [0, 2]:
                passed = min(1, self.car_counts[i])
                self.car_counts[i] -= passed
                self.wait_times[i] = 0
                cars_passed += passed

        elif self.current_phase == 3:  # BD left
            for i in [1, 3]:
                passed = min(1, self.car_counts[i])
                self.car_counts[i] -= passed
                self.wait_times[i] = 0
                cars_passed += passed

        return cars_passed
