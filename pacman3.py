import gym
import numpy as np

# --------------------------------------------------------
# 1) RAM extraction function
# --------------------------------------------------------
def get_player_position(ram):
    """
    Attempt to extract Ms. Pac-Man's (x, y) from the 128-byte RAM vector.
    NOTE: These offsets (10 for x, 13 for y) may not be correct in all
    MsPacman ROM versions; adjust if you see odd behavior.
    """
    x = ram[10]
    y = ram[13]
    
    # Ms. Pac-Man screen: 160 wide (0..159), 210 tall (0..209)
    x = max(0, min(x, 159))
    y = max(0, min(y, 209))
    return (x, y)

# --------------------------------------------------------
# 2) Discretization function
# --------------------------------------------------------
def discretize_state(player_pos, bins, limits):
    """
    Convert (x, y) into discrete indices so they can be used in the Q-table.
    bins: list like [num_bins_x, num_bins_y]
    limits: a list of (low, high), e.g. [(0,159), (0,209)]
    """
    discrete_coords = []
    for i, val in enumerate(player_pos):
        low, high = limits[i]
        edges = np.linspace(low, high, bins[i] + 1)
        idx = np.digitize(val, edges) - 1
        idx = np.clip(idx, 0, bins[i] - 1)
        discrete_coords.append(idx)
    return tuple(discrete_coords)

# --------------------------------------------------------
# 3) Epsilon-greedy action selection
# --------------------------------------------------------
def choose_action(state, q_table, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(q_table[state])

# --------------------------------------------------------
# 4) Custom (shaped) reward (optional)
# --------------------------------------------------------
def compute_shaped_reward(original_reward, info):
    """
    Example shaping: penalize losing a life by -50.
    Adjust as needed, or just return original_reward.
    """
    if "ale.lives" in info and info["ale.lives"] < 3:
        return original_reward - 50.0
    else:
        return original_reward

# --------------------------------------------------------
# 5) Training loop
# --------------------------------------------------------
def train_agent(n_episodes=10):
    # Use the RAM version of Ms. Pac-Man
    env = gym.make("ALE/MsPacman-ram-v5")

    n_actions = env.action_space.n
    print(f"Action space size: {n_actions}")

    # We'll discretize (x, y) into 20 bins each
    state_bins = [20, 20]
    # Ms. Pac-Man screen: x in [0..159], y in [0..209]
    state_limits = [(0, 159), (0, 209)]

    # Q-table shape: (20, 20, n_actions)
    q_table = np.zeros((*state_bins, n_actions))

    # Hyperparameters
    alpha = 0.1         # Learning rate
    gamma = 0.9         # Discount factor
    epsilon = 1.0       # Initial exploration
    epsilon_decay = 0.995
    epsilon_min = 0.1

    best_q_table = None
    best_score = float("-inf")

    for episode in range(n_episodes):
        # In gym 0.25.2, reset() returns ONLY obs
        obs = env.reset()
        info = {}

        player_pos = get_player_position(obs)
        state = discretize_state(player_pos, state_bins, state_limits)

        total_reward = 0.0
        done = False

        while not done:
            # 1) Choose an action
            action = choose_action(state, q_table, epsilon, n_actions)
            
            # 2) Step in environment
            next_obs, reward, done, info = env.step(action)

            # 3) Next state
            next_player_pos = get_player_position(next_obs)
            next_state = discretize_state(next_player_pos, state_bins, state_limits)

            # 4) (Optional) shaped reward
            shaped_reward = compute_shaped_reward(reward, info)

            # 5) Q-learning update
            best_next_action = np.argmax(q_table[next_state])
            td_target = shaped_reward + gamma * q_table[next_state][best_next_action]
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            # 6) Transition
            state = next_state
            total_reward += reward

        # Decrease epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Track best score
        if total_reward > best_score:
            best_score = total_reward
            best_q_table = np.copy(q_table)

        if episode % 100 == 0:
            print(f"Episode {episode}/{n_episodes}, total_reward={total_reward}, best_score={best_score:.1f}")

    env.close()
    print("Training finished!")
    return best_q_table, state_bins, state_limits

# --------------------------------------------------------
# 6) Testing the best agent
# --------------------------------------------------------
def test_agent(best_q_table, state_bins, state_limits, n_episodes=1):
    """
    Run test episodes in MsPacman (RAM) with the best Q-table.
    Renders to screen (human) so you can watch.
    """
    env = gym.make("ALE/MsPacman-ram-v5", render_mode="human")
    n_actions = env.action_space.n

    def best_action(state):
        return np.argmax(best_q_table[state])

    for ep in range(n_episodes):
        obs = env.reset()
        info = {}
        done = False
        total_reward = 0.0

        s = discretize_state(get_player_position(obs), state_bins, state_limits)

        while not done:
            # Choose best action from Q-table
            a = best_action(s)
            next_obs, reward, done, info = env.step(a)

            s = discretize_state(get_player_position(next_obs),
                                 state_bins, state_limits)
            total_reward += reward

        print(f"[Test Episode {ep}] Total Reward = {total_reward}")

    env.close()

# --------------------------------------------------------
# 7) Main entry point
# --------------------------------------------------------
if __name__ == "__main__":
    best_q, bins, limits = train_agent(n_episodes=10)

    print("\nTesting the best learned Q-table...")
    test_agent(best_q, bins, limits, n_episodes=3)

