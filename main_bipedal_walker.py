import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import random
import os
import json
from datetime import datetime
from gymnasium.wrappers import RecordVideo

from utils import plot_rewards, save_model, plot_losses, plot_steps, convert_to_json
from sac import SAC
from replay_buffer import ReplayBuffer
from config import config

# hyperparameters
num_episodes = 500  # total number of episodes to train
batch_size = 128  # batch size for updates
replay_capacity = 200000  # maximum transitions stored in the replay buffer

def train_one_agent(seed, num_episodes=num_episodes, batch_size=batch_size, replay_capacity=replay_capacity, config=config):
    """
    Trains a single SAC agent in the BipedalWalker-v3 environment.

    Parameters:
    ----------
    seed : int
        Random seed for reproducibility.
    num_episodes : int
        Total number of episodes to train.
    batch_size : int
        Batch size for updates.
    replay_capacity : int
        Size of the replay buffer.
    config : dict
        Hyperparameters for the SAC agent.

    Returns:
    -------
    episode_rewards : list
        List of total rewards per episode.
    """
    # set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # (optional: for plotting later) lists track losses during training
    # critic1_losses, critic2_losses, actor_losses, alpha_losses = [], [], [], []

    # (optional): track best performing episode
    # best_episode = -1
    # best_reward = 300    # threshold reward at which the env is considered solved

    # create environment and set seed
    env = gym.make("BipedalWalker-v3")
    env.reset(seed=seed)
    env.action_space.seed(seed)

    # get dimensions of the state and action spaces
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # get action bounds (assumes symmetric bounds)
    low = env.action_space.low
    high = env.action_space.high
    action_bounds = (low, high)

    # choose device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # initialise the agent and replay buffer
    agent = SAC(state_dim, action_dim, action_bounds, device, config)
    replay_buffer = ReplayBuffer(state_dim, action_dim, replay_capacity)

    episode_rewards = []  # stores the total rewards for each episode

    # main training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        # run one episode
        while not done:
            action = agent.select_action(state)  # select action using the current policy
            
            next_state, reward, terminated, truncated, _ = env.step(action)  # take action in the environment and observe the next state and reward
            done = terminated or truncated  # done is True if the episode is over

            total_reward += reward
            step += 1

            replay_buffer.push(state, action, reward, next_state, done)  # store the experience in the replay buffer

            state = next_state

            # update the agent once the replay buffer has enough samples
            if len(replay_buffer) > batch_size:
                for _ in range(2):  # 2 critic updates per actor update
                    critic_1_loss, critic_2_loss = agent.update_critic_only(replay_buffer, batch_size)
                    # (optional): critic1_losses.append(critic_1_loss)
                    # (optional): critic2_losses.append(critic_2_loss)

                actor_loss, alpha_loss = agent.update_actor_alpha(replay_buffer, batch_size)
                # (optional): actor_losses.append(actor_loss)
                # (optional): alpha_losses.append(alpha_loss)

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes} | Reward: {total_reward:.2f} | Steps: {step}")

        # (optional): save model if it beats the best_reward so far
        # if total_reward > best_reward:
        #     best_reward = total_reward
        #     best_episode = episode + 1
        #     save_model(agent, f"best_normalbiped_episode_{best_episode}")
        #     print(f"New best model saved at Episode {best_episode} with Reward {best_reward:.2f}")

        # (optional): save every 100 episodes for backup
        # if (episode + 1) % 100 == 0:
        #     save_model(agent, f"normalbiped_episode_{episode + 1}")

        # save final model at the end of training
        if episode == num_episodes - 1:
            save_model(agent, f"sac_final_seed_{seed}")
            print(f"Final model saved at Episode {episode + 1}")

    # (optional): after training, plot and save the rewards.
    # plot_rewards(episode_rewards)
    # plot_losses(critic1_losses, critic2_losses, actor_losses, alpha_losses)

    # save training summary as json
    summary = {
        "environment": "BipedalWalker-v3",
        # (optional): "best_episode": best_episode,
        # (optional): "best_reward": best_reward,
        "average_reward": float(np.mean(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "reward_curve": episode_rewards,
        "num_episodes": num_episodes,
        "batch_size": batch_size,
        "replay_capacity": replay_capacity,
    }

    json_path = f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Training summary saved to: {json_path}")

    env.close()
    return episode_rewards

def main():
    """
    Main function to train multiple SAC agents with different seeds
    and save their reward histories for later analysis.
    """
    all_rewards = []  # list to store rewards from all agents

    for run_id in range(5):  # train 5 agents
        print(f"Training agent {run_id + 1}/5...")
        seed = 42 + run_id  # different seed for each agent
        rewards = train_one_agent(seed)
        all_rewards.append(rewards)

    # save all agents reward histories as a json
    convert_to_json(all_rewards, agent_name="SAC_normal", file_name="sac_rewards_database")

if __name__ == "__main__":
    main()
