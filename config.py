# SAC hyperparameters
config = {
    'hidden_dim': 256,
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    'alpha_lr': 3e-4,
    'tau': 0.005,  # soft target update parameter
    'gamma': 0.99,  # discount factor for future rewards
    'target_entropy': -4,  # target entropy being the same as the number of actions is a common choice
}
