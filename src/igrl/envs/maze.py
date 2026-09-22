from dataclasses import dataclass
import numpy as np


@dataclass
class Position:
    row: int
    col: int


class MazeEnv:
    def __init__(self, max_steps: int = 100):
        self.grid = np.array([
                            [0, 0, 0, 1, 0, 0],
                            [1, 1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 0, 1],
                            [0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0]
                            ], dtype=np.int8,
                        ) # wall indentify by 1 and ways with 0

        self.actions = {
                        0: (-1, 0),
                        1: (1, 0),
                        2: (0, -1),
                        3: (0, 1),
                    }
        
        self.start = Position(0, 0)
        self.goal = Position(4, 5)

        self.max_steps = max_steps
        self.state = self.start
        self.steps = 0

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def shape(self):
        return self.grid.shape

    def reset(self):
        self.state = self.start
        self.steps = 0
        return self.state

    def step(self, action: int):
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")

        r, c = self.actions[action]
        temp_pos = Position(self.state.row+r, self.state.col+c,)
        if self._is_valid(temp_pos):
            self.state = temp_pos

        self.steps += 1
        reached_goal = self.state == self.goal
        timed_out = self.steps >= self.max_steps

        reward = 1.0 if reached_goal else 0.0
        last_move = reached_goal or timed_out

        return self.state, reward, last_move

    def _is_valid(self, position):
        row, col = position.row, position.col
        in_maze = ( (0 <= row < self.grid.shape[0] )
                       and (0 <= col < self.grid.shape[1]) )

        if not in_maze: 
            return False

        return self.grid[row, col] != 1 # hit the walls