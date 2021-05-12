import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        """
            Assert action is discrete.

        Args:
            obs (np.ndarray): size [N, ob_dim]

        Returns:
            actions (np.ndarray): size [N, ]

        """
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        # DONE: return the action that maxinmizes the Q-value
        # at the current observation as the output
        assert self.critic
        q_values_na = self.critic.qa_values(observation)

        actions = np.argmax(q_values_na, axis=-1).squeeze()

        return actions
