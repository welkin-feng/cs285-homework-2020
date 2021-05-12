import numpy as np
import time
import copy
import functools
import queue
import multiprocessing
import threading

############################################
############################################


def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)['observation']

    # predicted
    ob = np.expand_dims(true_states[0], 0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac, 0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states


def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def mean_squared_error(a, b):
    return np.mean((a - b)**2)

############################################
############################################


class Value:
    def __init__(self, value):
        self._value = value
        self.sem = threading.Semaphore(value=3)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.sem.acquire()
        self._value = value
        self.sem.release()


def parallel_sample_trajectory(args, paths, batch_size, lock=None):
    while True:
        path = sample_trajectory(*args)
        # lock.acquire()
        if sum(get_pathlength(p) for p in paths) < batch_size:
            paths.append(path)
        # lock.release()
        if sum(get_pathlength(p) for p in paths) >= batch_size:
            break


def parallel_sample_trajectory_q(args, result_q, batch_size, current_bs=None):
    while True:
        path = sample_trajectory(*args)
        result_q.put(path)
        if current_bs is not None:
            current_bs.value += get_pathlength(path)
            if current_bs.value >= batch_size:
                break


def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    # DONE: get this from hw1
    # initialize env for the beginning of a new rollout
    ob = env.reset()  # HINT: should be the output of resetting the env

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    image_obs.append(env.sim.render(
                        camera_name='track', height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob)  # HINT: query the policy's get_action function
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # DONE end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        rollout_done = done or steps >= max_path_length  # HINT: this is either 0 or 1
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length,
                        render=False, render_mode=('rgb_array')):
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.

        DONE implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    # DONE: get this from hw1
    parallel_mode = 1

    timesteps_this_batch = 0
    paths = []
    max_q_size = multiprocessing.cpu_count()

    if parallel_mode == 1:

        # multi-process: using Pool, keep sequential results by queue, this mode is ~1.5x
        # slower than mode 2, because main process needs to wait when getting async item in
        # queue, but it is accurate.
        # Bug: main process might be blocked by queue.get.
        # result_q = queue.Queue(max_q_size - 1)
        with multiprocessing.Pool(max_q_size) as pool:
            while timesteps_this_batch < min_timesteps_per_batch:
                func = functools.partial(sample_trajectory, env, policy,
                                         render=render, render_mode=render_mode)
                # jobs = pool.map_async(func, (max_path_length for _ in range(max_q_size)))
                for path in pool.map(func, [max_path_length] * max_q_size):
                    timesteps_this_batch += get_pathlength(path)
                    paths.append(path)
                    if timesteps_this_batch >= min_timesteps_per_batch:
                        break
                # while not result_q.full():
                #     try:
                #         result_q.put(pool.apply_async(sample_trajectory,
                #                                       (env, policy, max_path_length, render,
                #                                        render_mode)))
                #     except Exception:
                #         break
                # if not result_q.empty():
                #     try:
                #         path = result_q.get(block=False).get()
                #         timesteps_this_batch += get_pathlength(path)
                #         paths.append(path)
                #     except Exception:
                #         pass

    elif parallel_mode == 2:

        # multi-process: need to kill sub processes by main process cause sub processes may block.
        # Bug: the results of each process are the same.
        # multiprocessing.get_context('spawn')
        parallel_class = multiprocessing.Process
        result_q = multiprocessing.Queue(max_q_size)
        current_bs = multiprocessing.Value('i', 0)

        # multi-thread
        # Warning: gym may block by thread, multi threads may cause Error and then died.
        # parallel_class = threading.Thread
        # result_q = queue.Queue(max_q_size)
        # current_bs = Value(0)

        # multi-process or multi-thread
        tasks = []
        for i in range(max_q_size):
            # cannot sycn paths in multi process
            task = parallel_class(target=parallel_sample_trajectory_q,
                                  args=((env, policy, max_path_length, render, render_mode),
                                        result_q, min_timesteps_per_batch, current_bs))
            tasks.append(task)
            task.start()

        while timesteps_this_batch < min_timesteps_per_batch:
            path = result_q.get()
            timesteps_this_batch += get_pathlength(path)
            paths.append(path)

        try:
            result_q.close()
        except Exception:
            pass
        for t in tasks:
            try:
                t.terminate()
                t.close()
            except Exception:
                pass
            try:
                t.join()
            except Exception:
                pass
    elif parallel_mode == 3:
        # deprecated

        # multi-process: manager.list only can be used in Process but is slower than queue in mode 1
        # Bug: the results of each process are the same.
        # _ = multiprocessing.get_context('spawn')
        parallel_class = multiprocessing.Process
        lock = multiprocessing.Lock()
        paths = multiprocessing.Manager().list()

        # multi-thread: can sync `paths` between threads, but gym may block by thread.
        # Warning: multi threads may cause Error in gym and then die.
        # parallel_class = threading.Thread
        # lock = threading.Lock()

        # both multi-process and multi-thread are deprecated
        tasks = []
        for _ in range(max_q_size):
            task = parallel_class(target=parallel_sample_trajectory,
                                  args=((env, policy, max_path_length, render, render_mode),
                                        paths, min_timesteps_per_batch, lock))
            tasks.append(task)
            task.start()
        for t in tasks:
            t.join()
        paths = list(paths)

    else:
        while timesteps_this_batch < min_timesteps_per_batch:
            path = sample_trajectory(env, policy, max_path_length, render, render_mode)
            timesteps_this_batch += get_pathlength(path)
            paths.append(path)

    timesteps_this_batch = sum(get_pathlength(p) for p in paths)

    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length,
                          render=False, render_mode=('rgb_array')):
    """
        Collect ntraj rollouts.

        DONE implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
    """
    # DONE: get this from hw1
    paths = []
    for _ in range(ntraj):
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
    return paths


############################################
############################################


def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation": np.array(obs, dtype=np.float32),
            "image_obs": np.array(image_obs, dtype=np.uint8),
            "reward": np.array(rewards, dtype=np.float32),
            "action": np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return (observations,
            actions,
            next_observations,
            terminals,
            concatenated_rewards,
            unconcatenated_rewards)

############################################
############################################


def get_pathlength(path):
    return len(path["reward"])


def normalize(data, mean, std, eps=1e-8):
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    return data * std + mean


def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp)  # (num data points, dim)

    # mean of data
    mean_data = np.mean(data, axis=0)

    # if mean is 0,
    # make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    # width of normal distribution to sample noise from
    # larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data
