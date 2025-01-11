import numpy as np
import gym
import os

# Setează calea pentru ROM-uri
#os.environ["ALE_ROMS"] = "/usr/local/lib/python3.10/dist-packages/AutoROM/roms"

env = gym.make("ALE/MsPacman-v5", render_mode=None)

n_actions = env.action_space.n  # Numărul de acțiuni posibile
state_bins = [10, 10]  # Discretizarea simplificată pentru poziție
q_table = np.zeros((*state_bins, n_actions))  # Matricea Q
epsilon = 1.0  # Explorare inițială
epsilon_decay = 0.995  # Rata de scădere a explorării
epsilon_min = 0.1  # Explorare minimă
alpha = 0.1  # Rata de învățare
gamma = 0.9  # Factorul de discount
n_episodes = 100  # Număr de episoade pentru antrenament

# Extragem limitele de poziție din joc (arbitrare)
state_limits = [
    (0, 210),  # Poziția pe axa X
    (0, 160),  # Poziția pe axa Y
]

# Funcție pentru discretizarea poziției jucătorului
def discretize_state(player_pos, bins):
    """Discretizează poziția jucătorului într-o stare discretă."""
    discretized = []
    for i, s in enumerate(player_pos[:2]):  # Folosim poziția X și Y
        low, high = state_limits[i]
        bins_range = np.linspace(low, high, bins[i] + 1)
        index = np.digitize(s, bins_range) - 1
        discretized.append(np.clip(index, 0, bins[i] - 1))  # Asigurăm validitatea indexului
    return tuple(discretized)

# Funcție pentru alegerea unei acțiuni (epsilon-greedy)
def choose_action(state, q_table, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)  # Explorare
    return np.argmax(q_table[state])  # Exploatare

# Funcție de recompensare
def compute_reward(info, reward):
    """Calculăm recompensele ajustate."""
    if "ale.lives" in info and info["ale.lives"] < 3:  # Dacă agentul pierde o viață
        return -100
    elif reward > 0:  # Dacă colectează puncte
        return reward + 10
    else:  # Penalizăm staționarea
        return reward - 1

# Antrenarea agentului
best_q_table = None
best_score = float('-inf')

for episode in range(n_episodes):
    obs = env.reset()  # Observația inițială
    player_pos = obs[:2]  # Extragem poziția jucătorului
    state = discretize_state(player_pos, state_bins)
    total_reward = 0
    done = False

    while not done:
        action = choose_action(state, q_table, epsilon)
        next_obs, reward, done, info = env.step(action)  # Ajustăm pentru Gym 0.25.2
        next_player_pos = next_obs[:2]
        next_state = discretize_state(next_player_pos, state_bins)

        # Calculăm recompensa ajustată
        adjusted_reward = compute_reward(info, reward)

        # Actualizare Q-Table
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] += alpha * (
            adjusted_reward + gamma * q_table[next_state][best_next_action] - q_table[state][action]
        )
        state = next_state
        total_reward += adjusted_reward

    # Reducem explorarea
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Salvăm agentul cu cel mai mare scor
    if total_reward > best_score:
        best_score = total_reward
        best_q_table = np.copy(q_table)

    if episode % 100 == 0:
        print(f"Episode {episode}: Total Reward = {total_reward}, Best Score = {best_score}")

env.close()
print("Antrenare completă!")

# Testarea agentului cel mai bine antrenat
env = gym.make("ALE/MsPacman-v5", render_mode="human")
obs = env.reset()  # Observația inițială
player_pos = obs[:2]
state = discretize_state(player_pos, state_bins)
done = False
total_reward = 0

print("Testarea agentului cel mai bine antrenat începe. Urmărește jocul în timp real!")

while not done:
    action = np.argmax(best_q_table[state])  # Alege acțiunea cea mai bună
    next_obs, reward, done, info = env.step(action)
    next_player_pos = next_obs[:2]
    next_state = discretize_state(next_player_pos, state_bins)
    state = next_state
    total_reward += reward

env.close()
print(f"Scor total al agentului cel mai bine antrenat: {total_reward}")