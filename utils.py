import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import uuid
import json

def plot_rewards(rewards, filename="rewards_plot.png"):
    plt.figure(figsize=(10,6))
    plt.plot(rewards, linewidth=1)
    plt.xlabel("Episode Number", fontsize=12)
    plt.ylabel("Episode Return", fontsize=12)
    plt.title("SAC Agent Performance on BipedalWalker-v3", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Reward plot saved as {filename}")

def save_model(agent, filename):
    """Save the state dictionaries for actor and both critics."""
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_1_state_dict': agent.critic_1.state_dict(),
        'critic_2_state_dict': agent.critic_2.state_dict()
    }, filename.pt)
    print(f"Model saved to {filename}.pt")

def plot_losses(critic1, critic2, actor, alpha, filename="losses_plot.png"):
    """
    Plot SAC losses.

    Parameters:
    -----------
    critic1, critic2, actor, alpha : lists
        Losses logged per training step.

    filename : str
        Name of file to save plot.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(critic1, label="Critic 1 Loss")
    axs[0, 0].set_title("Critic 1 Loss")

    axs[0, 1].plot(critic2, label="Critic 2 Loss", color='orange')
    axs[0, 1].set_title("Critic 2 Loss")

    axs[1, 0].plot(actor, label="Actor Loss", color='green')
    axs[1, 0].set_title("Actor Loss")

    axs[1, 1].plot(alpha, label="Alpha Loss", color='red')
    axs[1, 1].set_title("Alpha Loss")

    for ax in axs.flat:
        ax.set(xlabel='Training Steps', ylabel='Loss')
        ax.grid(alpha=0.3)

    plt.suptitle("SAC Losses over Time", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Loss plot saved as {filename}")

def plot_steps(steps, filename="steps_plot.png"):
    plt.figure(figsize=(10,6))
    plt.plot(steps, label="Steps per Episode", linewidth=1)
    plt.xlabel("Episode Number", fontsize=12)
    plt.ylabel("Steps Taken", fontsize=12)
    plt.title("Steps per Episode over Training", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Steps plot saved as {filename}")

def convert_to_json(reward_history_ls: list[list[float]], agent_name: str, file_name: str, merge: bool = False):
    if os.path.exists(f"{file_name}.json"):
        with open(f"{file_name}.json", "r") as f:
            data = json.load(f)
            if agent_name in data:
                if merge:
                    pass
                else:
                    raise ValueError(
                        f"Agent '{agent_name}' already exists in {file_name}.json. Set merge flag to append to agent")
            else:
                data[agent_name] = {}
    else:
        data = {agent_name: {}}

    clean_rewards = [
        [float(r) for r in reward_history]
        for reward_history in reward_history_ls
    ]
    for _, reward_history in enumerate(clean_rewards):
        data[agent_name][str(uuid.uuid4())] = reward_history

    with open(f"{file_name}.json", "w") as f:
        json.dump(data, f, indent=4)
