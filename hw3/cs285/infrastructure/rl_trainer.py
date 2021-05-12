from collections import OrderedDict
import pickle
import os
import sys
import time

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger, get_logger, print_log

from cs285.agents.dqn_agent import DQNAgent
from cs285.infrastructure.dqn_utils import (
    get_wrapper_by_name,
    register_custom_envs,
)

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params):

        #############
        # INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])
        # Customize: logger and print_log
        get_logger(os.path.join(self.params['logdir'], "log.txt"))

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        # ENV
        #############

        # Make the gym environment
        register_custom_envs()
        self.env = gym.make(self.params['env_name'])
        if 'env_wrappers' in self.params:
            # These operations are currently only for Atari envs
            self.env = wrappers.Monitor(
                self.env,
                os.path.join(self.params['logdir'], "gym"),
                force=True,
                video_callable=(None if self.params['video_log_freq'] > 0 else False),
            )
            self.env = params['env_wrappers'](self.env)
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')
        if 'non_atari_colab_env' in self.params and self.params['video_log_freq'] > 0:
            self.env = wrappers.Monitor(
                self.env,
                os.path.join(self.params['logdir'], "gym"),
                force=True,
                video_callable=(None if self.params['video_log_freq'] > 0 else False),
            )
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')

        self.env.seed(seed)

        # import plotting (locally if 'obstacles' env)
        if not(self.params['env_name'] == 'obstacles-cs285-v0'):
            import matplotlib
            matplotlib.use('Agg')

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes

        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1 / self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30  # This is not actually used when using the Monitor wrapper
        elif 'video.frames_per_second' in self.env.env.metadata.keys():
            self.fps = self.env.env.metadata['video.frames_per_second']
        else:
            self.fps = 10

        #############
        # AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

        ###########################
        # Customize: SAVE PARAMS
        ###########################

        self.save_params()

    # Customize: save_params
    def save_params(self):
        for k, v in self.params.items():
            print_log(f"{k}: {v}")

    def run_training_loop(
        self,
        n_iter,
        collect_policy,
        eval_policy,
        initial_expertdata=None,
        relabel_with_expert=False,
        start_relabel_with_expert=1,
        expert_policy=None
    ):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = 1000 if isinstance(self.agent, DQNAgent) else 1

        for itr in range(n_iter):
            if itr % print_period == 0:
                print_log("\n********** Iteration %i ************\n" % itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.log_metrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # collect trajectories, to be used for training
            if isinstance(self.agent, DQNAgent):
                # only perform an env step and add to replay buffer for DQN
                self.agent.step_env()
                paths, envsteps_this_batch, train_video_paths = None, 1, None
            else:
                batch_size = self.params['batch_size']
                if itr == 0:
                    batch_size = self.params['batch_size_initial']
                trajectories = self.collect_training_trajectories(
                    itr, initial_expertdata, collect_policy, batch_size)
                paths, envsteps_this_batch, train_video_paths = trajectories

            self.total_envsteps += envsteps_this_batch

            # Get from hw1: relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr >= start_relabel_with_expert and paths is not None:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print_log("Training agent...\n")
            all_logs = self.train_agent()

            # log/save
            if self.log_video or self.log_metrics:
                # perform logging
                print_log('Beginning logging procedure...\n')
                if isinstance(self.agent, DQNAgent):
                    self.perform_dqn_logging(all_logs)
                else:
                    self.perform_logging(itr, paths, eval_policy, train_video_paths, all_logs)

                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))

    ####################################
    ####################################

    def collect_training_trajectories(
            self,
            itr,
            initial_expertdata,
            collect_policy,
            num_transitions_to_sample,
            save_expert_data_to_disk=False):
        """
        :param itr (int): iteration index
        :param initial_expertdata (str):  path to expert data pkl file
        :param collect_policy (BasePolicy):  the current policy using which we collect data
        :param batch_size (int):  the number of transitions we collect
        :return:
            paths (list[np.ndarray]): a list trajectories
            envsteps_this_batch (int): the sum over the numbers of environment steps in paths
            train_video_paths (list[np.ndarray]): paths which also contain videos for visualization purposes
        """
        # DONE: get this from Piazza

        # DONE decide whether to load training data or use the current policy to collect more data
        # HINT: depending on if it's the first iteration or not, decide whether to either
        # (1) load the data. In this case you can directly return as follows
        # ``` return loaded_paths, 0, None ```
        # if your initial_expertdata is None, then you need to collect new trajectories at *every* iteration
        if iter == 0 and initial_expertdata is not None:
            with open(initial_expertdata, 'rb') as f:
                loaded_paths = pickle.load(f.read())
            return loaded_paths, 0, None

        # (2) collect `self.params['batch_size']` transitions

        # DONE collect `num_transitions_to_sample` samples to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']
        print_log("Collecting data to be used for training...\n")
        paths, envsteps_this_batch = utils.sample_trajectories(
            self.env, collect_policy, num_transitions_to_sample, self.params['ep_len'])

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.log_video:
            print_log('Collecting train rollouts to be used for saving videos...\n')
            # DONE look in utils and implement sample_n_trajectories
            train_video_paths = utils.sample_n_trajectories(
                self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        # TODO: get this from Piazza
        print_log('Training agent using sampled data from replay buffer...\n')
        train_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            # DONE sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self.params['train_batch_size']
            batch_data = self.agent.sample(self.params['train_batch_size'])
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = batch_data

            # DONE use the sampled data to train an agent
            # HINT: use the agent's train function
            # HINT: keep the agent's training log for debugging
            train_log = self.agent.train(ob_batch, ac_batch, re_batch,
                                         next_ob_batch, terminal_batch)
            train_logs.append(train_log)
        return train_logs

    # Get from hw1
    def do_relabel_with_expert(self, expert_policy, paths):
        print_log("\nRelabelling collected observations with labels from an expert policy...")

        # DONE relabel collected obsevations (from our policy) with labels from an expert policy
        # HINT: query the policy (using the get_action function) with paths[i]["observation"]
        # and replace paths[i]["action"] with these expert labels
        for i in range(len(paths)):
            paths[i]["action"] = expert_policy.get_action(paths[i]["observation"])

        return paths

        ####################################
        ####################################
    def perform_dqn_logging(self, all_logs):
        last_log = all_logs[-1]

        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(
                self.best_mean_episode_reward, self.mean_episode_reward)

        logs = OrderedDict()

        logs["Train_EnvstepsSoFar"] = self.agent.t
        print_log("Timestep %d" % (self.agent.t,))
        if self.mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
        print_log("mean reward (100 episodes) %f" % self.mean_episode_reward)
        if self.best_mean_episode_reward > -5000:
            logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
        print_log("best mean reward %f" % self.best_mean_episode_reward)

        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            print_log("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start

        logs.update(last_log)

        sys.stdout.flush()

        for key, value in logs.items():
            print_log('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        print_log('Done logging...\n\n')

        self.logger.flush()

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print_log("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
            self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            print_log('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(
                self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            # save train/eval videos
            print_log('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps,
                                            max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,
                                            max_videos_to_save=MAX_NVIDEO,
                                            video_title='eval_rollouts')

        #######################

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print_log('{} : {}'.format(key, value))
                # self.logger.log_scalar(value, key, itr)
            print_log('Done logging...\n\n')

            self.logger.flush()