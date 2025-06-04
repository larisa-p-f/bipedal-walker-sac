# Experience Replay Buffer

import numpy as np

class ReplayBuffer:
    """
    Replay Buffer for storing past transitions (experience tuples) to sample from during training.

    Stores (state, action, reward, next_state, done) for off-policy learning.

    Parameters
    ----------
    obs_dim : int
        Dimension of the observation space.
    action_dim : int
        Dimension of the action space.
    capacity : int
        Maximum number of transitions to store in the buffer.
    """

    def __init__(self, obs_dim, action_dim, capacity):
        self.capacity = capacity  # max number of transitions to store

        # buffers for each component of the transition
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((capacity,), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)  # episode end flags

        self.ptr = 0  # pointer to next insert position
        self.size = 0  # current size of buffer

    def push(self, obs, action, reward, next_obs, done):
        """
        Store a new transition in the buffer.

        Parameters
        ----------
        obs : np.array
            Current observation (state).
        action : np.array
            Action taken.
        reward : float
            Reward received.
        next_obs : np.array
            Next observation (state).
        done : bool
            Whether the episode ended (True = 1.0, False = 0.0)
        """
        # save transition at the current pointer position
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity  # move pointer forward and wrap around if full
        self.size = min(self.size + 1, self.capacity)  # increase size until capacity is reached

    def sample(self, batch_size):
        """
        Sample a batch of transitions randomly from the buffer.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        obs, action, reward, next_obs, done : np.arrays
            Arrays of sampled transitions (batch_size elements each)
        """
        idx = np.random.randint(0, self.size, size=batch_size)  # randomly sample indices

        return (self.obs_buf[idx],
                self.action_buf[idx],
                self.reward_buf[idx],
                self.next_obs_buf[idx],
                self.done_buf[idx])

    def __len__(self):
        return self.size  # return current number of stored transitions