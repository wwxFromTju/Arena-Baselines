from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env import VecEnv
from multiprocessing import Process, Pipe
from gym_unity.envs import UnityEnv
import gym
import numpy as np
from .utils import clear_port, clear_ports
from .envs import TransposeImage, ExtraTimeLimit

try:
    from mpi4py import MPI
    print('# INFO: Using MPI')
except ImportError:
    print('# INFO: No MPI')
    MPI = None

def get_env_directory(env_name):
    import platform
    return './Bin/{}-{}'.format(
        env_name,
        platform.system(),
    )

class MultiAgentObservation(gym.ObservationWrapper):
    """For multiagent settings, convert between array and list"""
    def observation(self, observation):
        if len(observation)==1:
            # visual obs (this is ugly from unity, not my bad)
            return observation[0]
        else:
            # ram obs, [array(256), array(256), ...] -> array(num_agents, 156, )
            return np.stack(observation)

class MultiAgentReward(gym.RewardWrapper):
    """For multiagent settings, convert between array and list"""
    def reward(self,reward):
        return np.asarray(reward)

class DoneWrapper(gym.Wrapper):
    def reset(self):
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, self.done(done), info

    def done(self, done):
        deprecated_warn_once("%s doesn't implement 'done' method. Maybe it implements deprecated '_done' method." % type(self))
        return self._done(done)

class MultiAgentDone(DoneWrapper):
    """For multiagent settings, convert between array and list"""
    def done(self,done):
        return np.asarray(done)

class MultiAgentAction(gym.ActionWrapper):
    """For multiagent settings, convert between array and list"""
    def action(self,action):
        return list(action)

class MultiAgentRoller(gym.Wrapper):
    """roll the agent at different position of the env,
    in case some of the env may not be completely symmetric"""
    def __init__(self, env=None):
        super(MultiAgentRoller, self).__init__(env)
        self.shift = 0

    def reset(self):
        self.shift = np.random.randint(0, self.env.unwrapped.number_agents)
        return self.roll_back(self.env.reset())

    def roll(self, x):
        return np.roll(x, self.shift, axis=0)

    def roll_back(self, x):
        return np.roll(x, -self.shift, axis=0)

    def step(self, action):
        observation, reward, done, info = self.env.step(self.roll(action))
        return self.roll_back(observation), self.roll_back(reward), self.roll_back(done), info

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class MyUnityEnv(UnityEnv):

    def __init__(self, **args):
        super(MyUnityEnv, self).__init__(**args)
        self.train_mode = True

    def set_train_mode(self, train_mode):
        self.train_mode = train_mode

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        In the case of multi-agent environments, this is a list.
        Returns: observation (object/list): the initial observation of the
            space.
        """
        info = self._env.reset(train_mode=self.train_mode)[self.brain_name]
        n_agents = len(info.agents)
        self._check_agents(n_agents)
        self.game_over = False

        if not self._multiagent:
            obs, reward, done, info = self._single_step(info)
        else:
            obs, reward, done, info = self._multi_step(info)
        return obs


def worker(remote, parent_remote, env_name, max_episode_steps, port, use_visual):

    parent_remote.close()

    '''build env'''
    env = MyUnityEnv(
        environment_filename = get_env_directory(env_name),
        worker_id = port,
        use_visual = use_visual,
        multiagent = True,
    )

    env = MultiAgentObservation(env)
    env = MultiAgentReward(env)
    env = MultiAgentDone(env)
    env = MultiAgentAction(env)

    env = MultiAgentRoller(env)

    # # arena env runs in mutiple thread, the terminal condition may be trig for multiple times
    # # this is not dealt with in arena env, so deal with it here
    # env = ClipRewardEnv(env)
    # This issue no longer exists with the new framework, remove this.

    if max_episode_steps>0:
        env = ExtraTimeLimit(env, max_episode_steps=max_episode_steps)

    '''If the input has shape (W,H,3), wrap for PyTorch convolutions'''
    obs_shape = env.observation_space.shape
    if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        env = TransposeImage(env)

    def close():
        '''close this worker'''
        env.unwrapped.close()
        clear_port(port, ask=False)
        remote.close()
        print('# WARNING: Worker Closed')

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done.any():
                    ob = env.reset()
                if use_visual:
                    info['shift'] = env.env.env.shift
                else:
                    info['shift'] = env.env.shift
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'number_agents':
                remote.send(env.unwrapped.number_agents)
            elif cmd == 'close':
                close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'set_train_mode':
                env.unwrapped.set_train_mode(data)
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        close()
    finally:
        close()

class SubprocVecEnvUnity(SubprocVecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, num_env, env_name, max_episode_steps, start_index, use_visual, spaces=None):
        """
        Arguments:
        """
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_env)])
        self.ps = []
        rank = 0
        for (work_remote, remote) in zip(self.work_remotes, self.remotes):
            self.ps += [
                Process(
                    target=worker,
                    args=(work_remote, remote, env_name, max_episode_steps, (start_index+rank), use_visual)
                )
            ]
            rank += 1
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, num_env, observation_space, action_space)

    def number_agents(self):
        self._assert_not_closed()
        self.remotes[0].send(('number_agents', None))
        return self.remotes[0].recv()

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, list(actions)):
            remote.send(('step', action))
        self.waiting = True

    def set_train_mode(self, train_mode):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('set_train_mode', train_mode))

def make_arena(env_name, max_episode_steps, num_env, use_visual, start_index, device, gamma):

    clear_ports(start_index,num_env,ask=True)

    envs = SubprocVecEnvUnity(
        num_env=num_env,
        env_name = env_name,
        max_episode_steps = max_episode_steps,
        start_index=start_index,
        use_visual=use_visual,
    )
    from .envs import wrapper_envs_after_vec
    envs = wrapper_envs_after_vec(envs,device,gamma)
    return envs
