import abc
import itertools
import numpy as np
import torch
from torch import nn

from cs285.infrastructure import pytorch_util as ptu, utils
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = torch.optim.Adam(self.logits_na.parameters(),
                                              self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                          output_size=self.ac_dim,
                                          n_layers=self.n_layers,
                                          size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = torch.optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = torch.optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # DONE: get this from hw1
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # DONE return the action that the policy prescribes
        observation = ptu.from_numpy(observation.astype(np.float32))
        action_distribution = self(observation)
        action = action_distribution.sample()
        action = ptu.to_numpy(action)
        return action

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # DONE: get this from hw1
        # Hint:
        # For the discrete case, use categorical distribution.
        # For the continuous case, use multivariate gaussian.
        if self.discrete:
            logits_na = self.logits_na(observation)
            action_distribution = torch.distributions.Categorical(logits=logits_na)
        else:
            mean_na = self.mean_net(observation)
            std_na = torch.exp(self.logstd)
            action_distribution = torch.distributions.Normal(mean_na, std_na)
        return action_distribution


#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages, q_values=None):
        """
            Train step for each mlp network.
        Args:
            observations (np.ndarray): size [N, ob_dim]
            actions (np.ndarray): size [N,] (discrete) or [N, ac_dim] (continuous)
            advantages (np.ndarray): size [N,]
            q_values (np.ndarray): size [N,]
        """
        obs_no = ptu.from_numpy(observations)
        action_na = ptu.from_numpy(actions).view(obs_no.shape[0], -1)
        adv_n1 = ptu.from_numpy(advantages).view(obs_no.shape[0], 1)

        # DONE: compute the loss that should be optimized when training with policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
        # is the expectation over collected trajectories of:
        # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
        # by the `forward` method
        # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss

        action_distribution = self(obs_no)
        logprob_na = action_distribution.log_prob(action_na)
        loss = -(logprob_na * adv_n1).mean()

        # DONE: optimize `loss` using `self.optimizer`
        # HINT: remember to `zero_grad` first
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        train_log = {'Training Loss': ptu.to_numpy(loss), }

        if self.nn_baseline:
            # DONE: normalize the q_values to have a mean of zero and a standard deviation of one
            # HINT: there is a `normalize` function in `infrastructure.utils`
            q_values_norm = utils.normalize(q_values, q_values.mean(), q_values.std())
            q_n = ptu.from_numpy(q_values_norm)

            # DONE: use the `forward` method of `self.baseline` to get baseline predictions
            baseline_predictions_n = self.baseline(obs_no).squeeze()

            # avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
            # [ N ] versus shape [ N x 1 ]
            # HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1
            assert baseline_predictions_n.shape == q_n.shape

            # DONE: compute the loss that should be optimized for training the baseline MLP (`self.baseline`)
            # HINT: use `F.mse_loss`
            baseline_loss = self.baseline_loss(baseline_predictions_n, q_n)

            # DONE: optimize `baseline_loss` using `self.baseline_optimizer`
            # HINT: remember to `zero_grad` first
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

            train_log['Baseline Loss'] = ptu.to_numpy(baseline_loss)

        return train_log

    def run_baseline_prediction(self, obs):
        """
            Helper function that converts `obs` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `obs`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        obs = ptu.from_numpy(obs)
        predictions = self.baseline(obs)
        return ptu.to_numpy(predictions)[:, 0]
