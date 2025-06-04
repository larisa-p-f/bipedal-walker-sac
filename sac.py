# SAC Agent class (Actor, Critic, Alpha, Targets)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Actor(nn.Module):
    """
    The Actor network outputs the mean of the action distribution given an observation (state).

    Parameters:
    ----------
    obs_dim : int
        Dimension of the observation/state space from the environment.

    action_dim : int
        Dimension of the action space from the environment.

    hidden_dim : int
        Number of units in the hidden layers
    """
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        # define actor network
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim))
        
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs):
        """
        Forward pass through the actor network.

        Parameters:
        ----------
        obs : torch.Tensor
            The input observations (states) from the environment.
            Expected shape: (batch_size, obs_dim)
            
            - batch_size refers to the number of observations passed together.
            - During training (in update step), batch_size is typically set to config['batch_size'] (e.g., 256), 
            because we sample that many transitions from the replay buffer at once.
            - During environment interaction (select_action), batch_size is usually 1 because it's a single state.
        
        Returns:
        --------
        mean : torch.Tensor
            The mean of the action distribution for each observation. 
            Shape: (batch_size, action_dim)

        log_std : torch.Tensor
            The log of the standard deviation of the action distribution for each observation. 
            Shape: (batch_size, action_dim)
        """
        mean = self.net(obs)  # this is the mean of the distribution on the action space
        # log_std is a learnable parameter, initialised to 0, we find log std instead of std to avoid numerical instability
        log_std = self.log_std.expand_as(mean)  # log_std is the same for all actions in the batch, so we expand it to match the batch size
        return mean, log_std


    def sample(self, obs):
        """
        Samples an action from the actor's policy given an observation.

        This uses the reparameterisation trick:
        - Sample from Normal(mean, std)
        - Apply tanh to squash actions to (-1, 1)
        - Compute the corrected log probability
    
        Parameters:
        ----------
        obs : torch.Tensor
            Observation from the environment.
            Expected shape: (batch_size, obs_dim)

        Returns:
        --------
        action : torch.Tensor
            Squashed action sampled from the policy. 
            Shape: (batch_size, action_dim)

        log_prob : torch.Tensor
            Log probability of the sampled action (corrected for tanh squashing).
            Shape: (batch_size, 1)
        """
        mean, log_std = self.forward(obs)

        # clamping log_std prevents it from going to extreme values
        # log_std very small -> std almost 0 -> action almost deterministic -> no exploration
        # log_std very large -> std very large -> huge randomness, becomes just noise
        # std = exp(-20), almost 0 but not exactly 0, exp(2) -> reasonably large exploration
        log_std = torch.clamp(log_std, min=-20, max=2)

        std = log_std.exp()  # std is the exponential of log_std

        normal = torch.distributions.Normal(mean, std)  # create the normal distribution
        
        # reparameterisation trick
        x_t = normal.rsample()  # sample from the distribution, rsample allows gradients to flow through the sampling process
        raw_action = torch.tanh(x_t)  # squash the action to (-1, 1) using tanh
    
        # use log_prob because multiplying probs can cause instability
        log_prob = normal.log_prob(x_t)  # log probability of the sampled action
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # sum along action_dim (for multi-action environments like bipedal walker) -> we are summing here because these are LOG probs, for normal probs we would've multiplied
        log_prob -= torch.log(1 - raw_action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)  # correction for Tanh squashing

        return raw_action, log_prob



class Critic(nn.Module):
    """
    The Critic network estimates the Q-value for a given state-action pair.

    Parameters
    ----------
    obs_dim : int
        Dimension of the observation space from the environment.

    action_dim : int
        Dimension of the action space from the environment.

    hidden_dim : int
        Number of units in the hidden layers
    """

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        # input is state and action concatenated
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # outputs a single Q-value
        )

    def forward(self, obs, action):
        """
        Forward pass through the critic network.

        Parameters
        ----------
        obs : torch.Tensor
            Observation from the environment.
            Shape: (batch_size, obs_dim)

        action : torch.Tensor
            Action taken by the policy.
            Shape: (batch_size, action_dim)

        Returns
        -------
        q_value : torch.Tensor
            Estimated Q-value for the given state-action pair.
            Shape: (batch_size, 1)
        """
        # concatenate obs and action along the last dimension
        x = torch.cat([obs, action], dim=-1)
        q_value = self.net(x)
        return q_value


class SAC:
    """
    Soft Actor-Critic (SAC) Agent

    Parameters:
    ----------
    obs_dim : int
        Dimension of the observation space.
    action_dim : int
        Dimension of the action space.
    action_bounds : tuple
        Min and max values of action space.
    device : str
        'cpu' or 'cuda'.
    config : dict
        Hyperparameters.
    """
    def __init__(self, obs_dim, action_dim, action_bounds, device, config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.device = device
        self.config = config

        hidden_dim = config['hidden_dim']

        # create instances of all five networks
        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(device)  # actor network
        # two critic networks to mitigate overestimation bias of q-value
        self.critic_1 = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic_2 = Critic(obs_dim, action_dim, hidden_dim).to(device)
        # two target networks are used to stabilize training
        # they are copies of the critic networks, but their weights are updated slowly
        self.target_critic_1 = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.target_critic_2 = Critic(obs_dim, action_dim, hidden_dim).to(device)

        # copy initial weights to targets
        # without target networks, the same network would be used to predict current q-values and calculate target q-values
        # the target keeps moving while you're trying to learn it, leading to instability and divergence
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # optimisers
        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_1_optimiser = torch.optim.Adam(self.critic_1.parameters(), lr=config['critic_lr'])
        self.critic_2_optimiser = torch.optim.Adam(self.critic_2.parameters(), lr=config['critic_lr'])

        # entropy
        # alpha is the temperature parameter that controls the trade-off between exploration and exploitation
        # higher alpha -> more exploration (more random actions)
        # lower alpha -> more exploitation (more deterministic actions)
        # alpha is a learnable parameter, optimised along with the actor and critic networks
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)  # we are learning log_alpha and not alpha directly because we want alpha to always be positive, exp(log_alpha) is always positive
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=config['alpha_lr'])

        # target_entropy -> how much randomness we want to see in the policy
        # the default entropy if target_entropy is not defined is -action_dim (this default is common in the literature)
        # target_entropy closer to 0 -> very deterministic policy
        # target_entropy very negative -> very random policy
        self.target_entropy = config.get('target_entropy', -action_dim)

        # action scaling for env
        low, high = action_bounds
        self.action_scale = torch.FloatTensor((high - low) / 2.0).to(device)
        self.action_bias = torch.FloatTensor((high + low) / 2.0).to(device)


    def select_action(self, obs, evaluate=False):
        """
        Selects an action for the given observation.
        
        Parameters:
        ----------
        obs : np.array
            Observation from environment.
        evaluate : bool
            If True → deterministic action (mean of the policy distribution).
            If False → sample from action distribution (exploration).
        """
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        if evaluate:  # deterministic action
            mean, _ = self.actor.forward(obs)
            action = torch.tanh(mean)
        else:  # stochastic action
            action, _ = self.actor.sample(obs)

        # rescale action from (-1, 1) to action_bounds
        action = action * self.action_scale + self.action_bias
        action = action.detach().cpu().numpy()[0]
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        return action

    def update_critic_only(self, replay_buffer, batch_size):
        """
    Update only the critic networks (Q-functions) using sampled transitions.

    Parameters:
    ----------
    replay_buffer : ReplayBuffer
        Buffer containing previously observed transitions.

    batch_size : int
        Number of transitions to sample for the update.

    Returns:
    -------
    critic_1_loss : float
        Loss value for Critic 1 after the update step.

    critic_2_loss : float
        Loss value for Critic 2 after the update step.
    """
        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size)

        obs = torch.FloatTensor(obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        # compute target Q
        # we dont want to update the target critics with gradients, so we use torch.no_grad()
        # were only calculating a fixed target value (not learning from it)
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)  # use the actor to predict the next action and the log_prob of that action

            # target critics predict the Q-value for the next state and action
            target_Q1 = self.target_critic_1(next_obs, next_action)
            target_Q2 = self.target_critic_2(next_obs, next_action)

            # take the more pessimistic of the two Q-values (to avoid overestimation bias)
            target_Q = torch.min(target_Q1, target_Q2) - self.log_alpha.exp() * next_log_prob  # subtract entropy term (alpha * next_log_prob) to encourage exploration
            Q_target = reward + (1 - done) * self.config['gamma'] * target_Q  # this is the bellman backup target, if done = 1 episode is over, we dont want to add the next Q-value, otherwise if done = 0 we add the next Q-value discounted by gamma
            # (optional): clamp the target Q-value to avoid critic explosions -> smoother critic loss -> more stable learning
            # Q_target = torch.clamp(Q_target, -1000, 1000)  # might need lower clamp values than -1000 and 1000

        # current Q estimates
        Q1 = self.critic_1(obs, action)
        Q2 = self.critic_2(obs, action)

        # compute critic loss
        critic_1_loss = F.mse_loss(Q1, Q_target)
        critic_2_loss = F.mse_loss(Q2, Q_target)

        # update critic networks: clear old gradients, compute new gradients from loss, and apply updates
        # this allows the critics to learn better Q-value estimates over time
        self.critic_1_optimiser.zero_grad()
        self.critic_2_optimiser.zero_grad()
        (critic_1_loss + critic_2_loss).backward()

        self.critic_1_optimiser.step()
        self.critic_2_optimiser.step()

        # soft-update target critics
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.config['tau'] * param.data + (1 - self.config['tau']) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.config['tau'] * param.data + (1 - self.config['tau']) * target_param.data)

        return critic_1_loss.item(), critic_2_loss.item()
    
    def update_actor_alpha(self, replay_buffer, batch_size):
        """
        Update the actor (policy) network and the temperature parameter alpha.

        Parameters:
        ----------
        replay_buffer : ReplayBuffer
            Buffer containing previously observed transitions.

        batch_size : int
            Number of transitions to sample for the update.

        Returns:
        -------
        actor_loss : float
            Loss value for the actor after the update step.

        alpha_loss : float
            Loss value for alpha (temperature) after the update step.
        """
        # sample batch of states (ignore actions, rewards, etc.)
        states, *_ = replay_buffer.sample(batch_size)
        obs = torch.FloatTensor(states).to(self.device)

        new_action, log_prob = self.actor.sample(obs)  # sample new action from the actor's policy
        Q1_new = self.critic_1(obs, new_action)  # predict Q-value for the new action
        Q2_new = self.critic_2(obs, new_action)
        Q_new = torch.min(Q1_new, Q2_new)  # take the more pessimistic of the two Q-values (to avoid overestimation bias)

        actor_loss = (self.log_alpha.exp() * log_prob - Q_new).mean()  # actor loss: encourage high Q-values and high entropy (weighted by alpha)

        # update actor network
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()  # alpha loss: adjust temperature to match the target entropy

        # update alpha
        self.alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.alpha_optimiser.step()

        return actor_loss.item(), alpha_loss.item()





