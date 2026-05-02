"""Traffic timing and reward settings used by the DQN environment.

The values are intentionally plain dictionaries so experiments can tune the
simulation without editing the environment logic.
"""

DECISION_INTERVAL_SECONDS = 10
PHASE_DURATION_CHOICES_SECONDS = list(range(10, 100, 10))
TURN_ORDER = ["left", "straight", "right"]
MAX_EPISODE_SECONDS = 1800

PHASES = [
    {
        "name": "all_right",
        "approaches": [0, 1, 2, 3],
        "movement_capacities": {"right": 2},
        "description": "All right-turn lanes",
    },
    {
        "name": "AC_forward",
        "approaches": [0, 2],
        "movement_capacities": {"straight": 2, "right": 2},
        "description": "A/C straight and right-turn traffic",
    },
    {
        "name": "BD_forward",
        "approaches": [1, 3],
        "movement_capacities": {"straight": 2, "right": 2},
        "description": "B/D straight and right-turn traffic",
    },
    {
        "name": "AC_left",
        "approaches": [0, 2],
        "movement_capacities": {"left": 1},
        "description": "A/C protected left turns",
    },
    {
        "name": "BD_left",
        "approaches": [1, 3],
        "movement_capacities": {"left": 1},
        "description": "B/D protected left turns",
    },
]

MIN_GREEN_SECONDS = 10
TARGET_MAX_GREEN_SECONDS = 60

ARRIVAL_LOW_INCLUSIVE = 0
ARRIVAL_HIGH_EXCLUSIVE = 3
INITIAL_QUEUE_LOW_INCLUSIVE = 0
INITIAL_QUEUE_HIGH_EXCLUSIVE = 10

STATE_NORMALIZATION = {
    "queue": 20.0,
    "wait": 120.0,
    "phase_time": max(PHASE_DURATION_CHOICES_SECONDS),
}

REWARD_WEIGHTS = {
    "cars_passed": 5.0,
    "total_wait": -0.03,
    "total_queue": -0.5,
    "overtime": -4.0,
    "switch_before_min": -0.7,
    "starvation": -1.0,
    "served_pressure": 0.08,
    "pressure_gap": -0.18,
}
