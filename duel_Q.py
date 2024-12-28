import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Hyperparameters
STATE_SHAPE = (4, 84, 84)  # Channels-first for PyTorch
ACTION_SPACE = 6  # Number of actions in MsPacman
GAMMA = 0.99
LEARNING_RATE = 0.00025
BATCH_SIZE = 32
MEMORY_SIZE = 100000
UPDATE_TARGET_EVERY = 1000
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1
START_EPSILON = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DuelingDQN(nn.Module):
    def __init__(self, action_space):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)  # Reduced number of filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # Reduced number of filters
        self.fc = nn.Linear(32 * 432, 128)  # Reduced size of FC layer

        # Value stream
        self.value = nn.Linear(128, 1)
        # Advantage stream
        self.advantage = nn.Linear(128, action_space)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc(x))

        value = self.value(x)
        advantage = self.advantage(x)

        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values



# Agent
class DuelingDQNAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = START_EPSILON
        self.steps = 0

        # Initialize networks
        self.model = DuelingDQN(action_space).to(DEVICE)
        self.target_model = DuelingDQN(action_space).to(DEVICE)
        self.update_target_network()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self):
        return random.sample(self.memory, BATCH_SIZE)

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = self.sample_memory()
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + (1 - dones) * GAMMA * next_q_values

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % UPDATE_TARGET_EVERY == 0:
            self.update_target_network()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

# Preprocessing functions
def preprocess_frame(frame):
    frame = np.mean(frame, axis=2).astype(np.uint8)  # Grayscale
    frame = frame / 255.0  # Normalize
    return frame

def stack_frames(stacked_frames, frame):
    frame = preprocess_frame(frame)
    if stacked_frames is None:
        stacked_frames = np.stack([frame] * STATE_SHAPE[0], axis=0)
    else:
        stacked_frames = np.append(stacked_frames[1:], frame[np.newaxis, ...], axis=0)
    return stacked_frames


env = gym.make("ALE/MsPacman-v5", render_mode=None)
agent = DuelingDQNAgent(env.action_space.n)  # Ensure this uses the smaller model defined above
# #Load the saved model if it exists
try:
    agent.model.load_state_dict(torch.load("dueling_dqn_model.pth"))
    agent.model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully!")
except FileNotFoundError:
    print("No saved model found, starting from scratch.")

EPISODES = 10
for episode in range(EPISODES):
    obs = env.reset()  # Get initial observation
    state = stack_frames(None, obs)  # Preprocess and stack frames
    total_reward = 0
    done = False
    tries = 100
    while not done:
        action = agent.act(state)
        next_frame, reward, done, info = env.step(action)  # Gym 0.25 returns 4 values
        next_state = stack_frames(state, next_frame)

        agent.store_transition(state, action, reward, next_state, done)
        agent.train()
        agent.decay_epsilon()
        tries -= 1
        state = next_state
        total_reward += reward
        if tries % 50 == 0:
            print(tries, total_reward)

    print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}")

torch.save(agent.model.state_dict(), "dueling_dqn_model.pth")
env.close()

print("Testing the agent's performance after training...")
env = gym.make("ALE/MsPacman-v5", render_mode="human")
test_episodes = 5  # Number of episodes to test
for test_episode in range(test_episodes):
    obs = env.reset()  # Get initial observation
    state = stack_frames(None, obs)  # Preprocess and stack frames
    total_reward = 0
    done = False
    tries = 0

    while not done:
        action = agent.act(state)  # Choose the best action from the trained agent
        next_frame, reward, done, info = env.step(action)  # Get the next state and reward

        next_state = stack_frames(state, next_frame)
        state = next_state
        total_reward += reward
        tries += 1

    print(f"Test Episode {test_episode + 1}/{test_episodes}, Total Reward: {total_reward}")

env.close()  # Close the environment after testing
# Save the model after training
