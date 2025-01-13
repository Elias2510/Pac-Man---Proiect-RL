import gym
from gym.wrappers import AtariPreprocessing, FrameStack
from stable_baselines3 import DQN
import numpy as np
import cv2

def watch_trained_agent(model_path="dqn_mspacman_model.zip", n_episodes=10):
    env_id = "MsPacmanNoFrameskip-v4"
    env = gym.make(env_id)

    # Match the training wrappers (downscale to 84, grayscale, frame skip, etc.)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=6, scale_obs=False)
    env = FrameStack(env, num_stack=4)  # Yields LazyFrames

    model = DQN.load(model_path)

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Convert LazyFrames -> np.array
            obs_array = np.array(obs)
            action, _ = model.predict(obs_array, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # --- Manual rendering in a bigger window ---
            # 1) Get the raw RGB frame:
            frame = env.render(mode="rgb_array")
            # 'frame' has shape (210, 160, 3) or similar (the actual game screen)
            
            # 2) Resize the frame to something bigger, e.g. 4x:
            big_frame = cv2.resize(frame, (160*4, 210*4), interpolation=cv2.INTER_NEAREST)
            # or you can pick any size you like, e.g. (640, 420)

            # 3) Show in an OpenCV window:
            cv2.imshow("MsPacman - Larger Window", big_frame)
            # Wait a bit so the window can update;
            # press 'q' to exit early:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                done = True

        print(f"[Episode {ep+1}] Reward: {total_reward}")

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    watch_trained_agent("dqn_mspacman_model.zip")

