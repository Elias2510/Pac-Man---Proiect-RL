import gym
import signal
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


# Variabilă globală pentru a putea fi accesată din signal handler
model = None

def signal_handler(sig, frame):
    """
    Callback apelat când se apasă Ctrl+C (SIGINT).
    """
    print("\n[INFO] CTRL+C detectat! Salvez modelul înainte de ieșire...")
    if model is not None:
        model.save("/home/alex/Desktop/RL/dqn_mspacman_model_interrupt")
        print("[INFO] Model salvat ca dqn_mspacman_model_interrupt.zip")
    sys.exit(0)

# Înregistrăm handler-ul pentru semnalul SIGINT
signal.signal(signal.SIGINT, signal_handler)


def train_dqn_on_mspacman(total_timesteps=100000):
    env_id = "MsPacmanNoFrameskip-v4"
    env = make_atari_env(env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)

    dqn_model = DQN(
        policy="CnnPolicy",
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
        train_freq=4,
    )

    print(f"Training DQN on MsPacman for {total_timesteps} timesteps...")
    dqn_model.learn(total_timesteps=total_timesteps)
    env.close()
    return dqn_model

def watch_trained_agent(trained_model, n_episodes=1):
    env_id = "MsPacmanNoFrameskip-v4"
    test_env = gym.make(env_id)

    for episode in range(n_episodes):
        obs = test_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            test_env.render()
            action, _ = trained_model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            total_reward += reward

        print(f"[Episode {episode+1}] Reward: {total_reward}")

    test_env.close()


def main():
    global model  # Ca să putem accesa și modifica variabila globală `model`
    # 1) Antrenăm modelul
    model = train_dqn_on_mspacman(total_timesteps=20)
    model.save("test_model")
    print("Am salvat un test_model.zip!")

    # 2) Salvăm modelul
    model.save("dqn_mspacman_model")
    print("Model salvat ca dqn_mspacman_model.zip")

    # 3) Afișăm câteva episoade pentru test
    watch_trained_agent(model, n_episodes=3)


if __name__ == "__main__":
    main()

