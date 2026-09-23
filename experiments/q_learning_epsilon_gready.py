import numpy as np
from igrl.envs.maze import MazeEnv
from igrl.agents.q_learning import QLearningAgent


env = MazeEnv(max_steps=100)

agent = QLearningAgent( maze_shape=env.env_grid.shape, n_actions=env.n_actions,
                       learning_rate=0.1, gamma=0.99, epsilon=0.1, seed=3456)


n_episodes = 2000
successes = []
episode_steps = []
for episode in range(n_episodes):
    state = env.reset()
    while True:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        agent.update( state=state, action=action, reward=reward,
                     next_state=next_state, done=done)

        state = next_state

        if done:
            successes.append(reward == 1.0)
            episode_steps.append(env.steps)
            break



action_symbols = {
                    0: "↑",
                    1: "↓",
                    2: "←",
                    3: "→",
                }

for row in range(env.env_grid.shape[0]):

    for col in range(env.env_grid.shape[1]):

        if env.grid[row, col] == 1:
            print("#", end=" ")

        elif (row, col) == (env.goal[0], env.goal[1]):
            print("G", end=" ")

        else:
            q_values = agent.q_table[row, col]
            
            if np.allclose(q_values, 0):
                print("?", end=" ")
            else:
                action = np.argmax(q_values)
                print(action_symbols[action], end=" ")
    
    print()

        