import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import numpy as np

def train_dqn_on_mspacman(total_timesteps=500000):
    """
    Train a DQN on Ms. Pac-Man for a specified number of timesteps,
    using Gym 0.21. Returns the trained model.
    """

    # 1) Environment ID for older Gym
    env_id = "MsPacmanNoFrameskip-v4"

    # 2) Create Atari env with standard wrappers
    #    n_envs=1 (a single environment), random seed=0 for reproducibility
    env = make_atari_env(env_id, n_envs=1, seed=0)

    # 3) Stack the last 4 frames so the agent has a sense of motion
    env = VecFrameStack(env, n_stack=4)

    # 4) Create the DQN model
    model = DQN(
        policy="CnnPolicy",    # Convolutional NN for image input
        env=env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=50000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        target_update_interval=1000,
        train_freq=4,   # Train every 4 steps
    )

    print(f"Training DQN on MsPacman for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    env.close()
    return model

def watch_trained_agent(model, n_episodes=1):
    """
    Run the trained model in Ms. Pac-Man with old Gym API (0.25.2),
    calling env.render() each step to visualize.
    """
    env_id = "MsPacmanNoFrameskip-v4"
    test_env = gym.make(env_id)  # No 'render_mode' in old Gym
    # test_env.reset() returns only obs

    for episode in range(n_episodes):
        obs = test_env.reset()  # Gym 0.25.2 returns just 'obs'
        done = False
        total_reward = 0.0

        while not done:
            # Render each step to watch in real time
            test_env.render()

            # Predict best action
            action, _ = model.predict(obs, deterministic=True)

            # Step the environment
            obs, reward, done, info = test_env.step(action)
            total_reward += reward

        print(f"[Episode {episode+1}] Reward: {total_reward}")

    test_env.close()

def main():
    # 1) Train the model (200k timesteps, for example)
    model = train_dqn_on_mspacman(total_timesteps=500000)

    # 2) Save the model
    model.save("dqn_mspacman_model")
    print("Model saved as dqn_mspacman_model.zip")

    # 3) (Optional) Load model
    # model = DQN.load("dqn_mspacman_model")

    # 4) Watch the trained agent
    watch_trained_agent(model, n_episodes=3)

if __name__ == "__main__":
    main()

