import numpy as np


class QLearningAgent:
    def __init__(self, maze_shape, n_actions,
                learning_rate = 0.1, gamma = 0.99,
                epsilon = 0.1, seed: int | None = None,
            ): # tuple[int, int], int, float, float, float, int
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros( (*maze_shape, n_actions), dtype=np.float64)

    def select_action(self, state): 
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        q_values = self.q_table[state[0], state[1]]
        best_actions = np.flatnonzero(q_values == q_values.max())

        return int(self.rng.choice(best_actions)) # must be int in n_actions

    def update(self, state, action, reward, next_state, done ): # Position, int, float, Position, bool
        q_s_a = self.q_table[state[0], state[1], action]

        if done:
            temp = reward
        else:
            temp = reward + self.gamma * np.max( self.q_table[next_state[0], next_state[1]] ) # changing  = alpha * [ r + gamma * max_action (s, a) ]

        self.q_table[state[0], state[1], action] += self.learning_rate * (temp - q_s_a)
    