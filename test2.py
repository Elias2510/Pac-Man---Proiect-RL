import gymnasium as gym

try:
    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
    print("MsPacman environment initialized successfully!")
    env.close()
except Exception as e:
    print(f"Error: {e}")

