import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import json
import time

from sac import SAC
from config import config

seed = int(time.time()) % (2**32 - 1)  # time-based seed
np.random.seed(seed)
torch.manual_seed(seed)
print(f"Randomly assigned seed: {seed}")


def evaluate_model(model_path, num_episodes=100):
    env = gym.make("BipedalWalker-v3")
    env.reset(seed=seed)
    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    low, high = env.action_space.low, env.action_space.high
    action_bounds = (low, high)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SAC(state_dim, action_dim, action_bounds, device, config)

    checkpoint = torch.load(model_path)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])

    print(f"Evaluating {model_path}...")

    episode_returns = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        

        while not done:
            action = agent.select_action(state, evaluate=True)  # deterministic policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            state = next_state

        print(f"Episode {ep+1:2d} | Return: {total_reward:.2f} | Steps: {step}")
        episode_returns.append(total_reward)

    env.close()

    average_return = np.mean(episode_returns)
    print(f"Average Return over {num_episodes} episodes: {average_return:.2f}")

    summary = {
        "model_path": model_path,
        "seed": seed,
        "num_episodes": num_episodes,
        "average_return": float(average_return)
    }

    # Save to JSON file
    summary_filename = f"evaluation_summary_seed_{seed}.json"
    with open(summary_filename, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved evaluation summary to {summary_filename}")

    # Plot total rewards
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, num_episodes + 1), episode_returns, linewidth=1)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('SAC Evaluation: Total Reward per Episode', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("evaluation_results.png")
    plt.close()
    print("Evaluation reward plot saved as evaluation_results_rewards_only.png")

if __name__ == "__main__":
    evaluate_model("sac_final_seed_42.pt")
