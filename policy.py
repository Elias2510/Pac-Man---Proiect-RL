import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Definim rețeaua neuronală pentru Policy Gradient
class PolicyNetworkCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetworkCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc_input_dim = 64 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Funcția de preprocesare a observațiilor
def preprocess_observation(obs):
    if isinstance(obs, np.ndarray) and len(obs.shape) == 3:  # Imagine RGB
        obs = np.mean(obs, axis=2)  # Convertim la grayscale
    obs = obs / 255.0  # Normalizare între 0 și 1
    obs = torch.tensor(obs, dtype=torch.float32)
    obs = obs.unsqueeze(0).unsqueeze(0)  # Adaugăm dimensiuni batch și canal
    obs = torch.nn.functional.interpolate(obs, size=(84, 84), mode='bilinear', align_corners=False)
    return obs

# Alegerea acțiunii bazată pe probabilități
def choose_action(policy_net, state):
    probs = policy_net(state)
    action = torch.multinomial(probs, num_samples=1).item()
    return action, torch.log(probs[0, action])

# Calcularea recompenselor cumulate (returnuri)
def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    return (returns - returns.mean()) / (returns.std() + 1e-9)

# Funcția pentru antrenare cu entropie și batch de antrenament
def train_agent(env, policy_net, optimizer, gamma, n_episodes, entropy_weight=0.01):
    for episode in range(n_episodes):
        obs = env.reset()
        state = preprocess_observation(obs[0])
        log_probs = []
        rewards = []
        total_reward = 0
        done = False

        while not done:
            action, log_prob = choose_action(policy_net, state)
            next_obs, reward, done, _= env.step(action)
            next_state = preprocess_observation(next_obs)

            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward

            state = next_state

        # Calcularea și normalizarea returnurilor
        returns = compute_returns(rewards, gamma)
        log_probs_tensor = torch.stack(log_probs)

        # Calcularea entropiei pentru îmbunătățirea explorării
        entropy = -torch.sum(torch.exp(log_probs_tensor) * log_probs_tensor)

        # Calcularea pierderii (adăugăm entropia)
        loss = -torch.sum(log_probs_tensor * returns) - entropy_weight * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Loss: {loss.item()}")

# Funcție pentru testare cu vizualizare pe mai multe episoade
def test_agent(env, policy_net, n_episodes):
    for episode in range(n_episodes):
        obs = env.reset()
        state = preprocess_observation(obs[0])
        total_reward = 0
        done = False

        while not done:
            action, _ = choose_action(policy_net, state)
            next_obs, reward, done, _= env.step(action)
            state = preprocess_observation(next_obs)
            total_reward += reward

        print(f"Scor total al agentului în episodul {episode + 1}: {total_reward}")

# Alegerea modului de rulare
if __name__ == "__main__":
    mode = input("Alege modul (train/test): ").strip().lower()

    env_name = "ALE/MsPacman-v5"
    gamma = 0.89
    n_episodes = 1000  # Poți modifica acest număr dacă vrei să testezi mai puține episoade
    n_test_episodes = n_episodes  # Folosește același număr de episoade pentru testare
    n_actions = gym.make(env_name).action_space.n
    input_channels = 1

    policy_net = PolicyNetworkCNN(input_dim=input_channels, output_dim=n_actions)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    if mode == "train":
        env = gym.make(env_name, render_mode=None)  # Mod non-grafic
        train_agent(env, policy_net, optimizer, gamma, n_episodes)
        env.close()
        print("Antrenare completă!")
    elif mode == "test":
        env = gym.make(env_name, render_mode="human")  # Vizualizare grafică
        test_agent(env, policy_net, n_test_episodes)  # Folosește aceleași episoade
        env.close()
        print("Testare completă!")
    else:
        print("Mod invalid! Alege 'train' sau 'test'.")
