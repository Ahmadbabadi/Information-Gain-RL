from dataclasses import dataclass
import numpy as np


class MazeEnv:
    def __init__(self, max_steps = 100):
        self.grid = np.array([
                            [0, 0, 0, 1, 0, 0],
                            [1, 1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 0, 1],
                            [0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0]
                            ], dtype=np.int8,
                        ) # wall indentify by 1 and ways with 0
        self.actions = np.array([
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1]
        ])
        self.start = np.array([0, 0])
        self.goal = np.array([4, 5])
        self.max_steps = max_steps
        self.state = self.start.copy()
        self.steps = 0

    @property
    def n_actions(self):
        return self.actions.shape[0]

    @property
    def env_grid(self):
        return self.grid

    def reset(self):
        self.state = self.start.copy()
        self.steps = 0
        return self.state

    def step(self, action):
        if not ( 0 <= action < self.actions.shape[0]):
            raise ValueError(f"Invalid action: {action}")

        temp_state = self.state + self.actions[action]
        if self._is_valid(temp_state):
            self.state = temp_state

        self.steps += 1
        reached_goal = np.all(self.state == self.goal)
        timed_out = self.steps >= self.max_steps
        reward = 1.0 if reached_goal else 0.0
        last_move = reached_goal or timed_out

        return self.state, reward, last_move

    def _is_valid(self, temp_state):
        in_maze = ( (0 <= temp_state[0] < self.grid.shape[0] )
                       and (0 <= temp_state[1] < self.grid.shape[1]) )
        if not in_maze: 
            return False
        return self.grid[temp_state[0], temp_state[1]] != 1 # hit the walls