import time
import os
import torch
import numpy as np
from .utils import update_linear_schedule
from .logger import EpisodeScalerSummary


class Agent(object):
    """docstring for Agent."""

    def __init__(self, id, envs, recurrent_brain, num_processes, num_steps, use_linear_lr_decay, use_linear_clip_decay, use_gae, gamma, tau, num_env_steps, num_updates, log_dir, tf_summary, cuda, device,
                 trainer_id, value_loss_coef, entropy_coef, lr, eps, alpha, max_grad_norm, clip_param, ppo_epoch, num_mini_batch, log_interval, vis, vis_interval):
        super(Agent, self).__init__()

        '''basic'''
        self.id = id
        self.envs = envs
        self.recurrent_brain = recurrent_brain
        self.num_processes = num_processes
        self.num_steps = num_steps
        self.use_linear_lr_decay = use_linear_lr_decay
        self.use_linear_clip_decay = use_linear_clip_decay
        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau
        self.num_env_steps = num_env_steps
        self.num_updates = num_updates
        self.log_dir = log_dir
        self.tf_summary = tf_summary
        self.cuda = cuda
        self.device = device

        '''brain'''
        self.trainer_id = trainer_id
        self.lr = lr
        self.clip_param = clip_param

        '''log'''
        self.log_interval = log_interval
        self.vis = vis
        self.vis_interval = vis_interval

        '''build policy'''
        base_kwargs = {'recurrent': self.recurrent_brain}
        if len(self.envs.observation_space.shape) == 1:
            # for ram obs, the hidden_size should be same as the obs size, but not smaller than 64
            base_kwargs['hidden_size'] = max(
                int(self.envs.observation_space.shape[0]), 64)
        from .brains import Policy
        self.brain = Policy(self.envs.observation_space.shape, self.envs.action_space,
                            base_kwargs=base_kwargs).to(self.device)

        self.update_i = 0
        self.num_trained_frames_start = self.update_i * \
            self.num_processes * self.num_steps

        '''build trainer'''
        if self.trainer_id == 'a2c':
            from .trainers import A2C_ACKTR
            self.trainer = A2C_ACKTR(self.brain, value_loss_coef,
                                     entropy_coef, lr=self.lr,
                                     eps=eps, alpha=alpha,
                                     max_grad_norm=max_grad_norm)
        elif self.trainer_id == 'ppo':
            from .trainers import PPO
            self.trainer = PPO(self.brain, self.clip_param, ppo_epoch, num_mini_batch,
                               value_loss_coef, entropy_coef, lr=self.lr,
                               eps=eps,
                               max_grad_norm=max_grad_norm)
        elif self.trainer_id == 'acktr':
            from .trainers import A2C_ACKTR
            self.trainer = A2C_ACKTR(self.brain, value_loss_coef,
                                     entropy_coef, acktr=True)

        '''build rollouts'''
        from assets.storage import RolloutStorage
        self.rollouts = RolloutStorage(self.num_steps, self.num_processes,
                                       self.envs.observation_space.shape, self.envs.action_space,
                                       self.brain.recurrent_hidden_state_size).to(self.device)

        self.episode_scaler_summary = EpisodeScalerSummary(['raw', 'len'])

        self.time_start = time.time()
        self.step_i = 0

        self.current_checkpoint_by_frame = 0

        if len(self.envs.observation_space.shape) == 1:
            # ram obs does not supporte test_obs
            self.test_obs = False
        else:
            # visual obs support test_obs, this will log the video of first episode since this run to tensorboard
            self.test_obs = True

        self.obs_video = None

    def reset(self, obs):
        self.rollouts.obs[0].copy_(obs)

    def schedule_trainer(self):
        '''update schedule'''
        if self.use_linear_lr_decay:
            '''decrease learning rate linearly'''
            if self.trainer_id == "acktr":
                '''use optimizer's learning rate since it's hard-coded in kfac.py'''
                update_linear_schedule(
                    self.trainer.optimizer, self.update_i, self.num_updates, self.trainer.optimizer.lr)
            else:
                update_linear_schedule(
                    self.trainer.optimizer, self.update_i, self.num_updates, self.lr)

        '''clip parameters'''
        if self.trainer_id == 'ppo' and self.use_linear_clip_decay:
            self.trainer.clip_param = self.clip_param * \
                (1 - self.update_i / float(self.num_updates))

    def act(self, obs, mode):

        self.test_obs_at_act(obs)

        if mode in ['playing']:
            deterministic = True
        elif mode in ['learning']:
            deterministic = False

        with torch.no_grad():
            obs, recurrent_hidden_states, masks = self.rollouts.get_policy_inputs(
                self.step_i)
            self.value, self.action, self.action_log_prob, self.recurrent_hidden_states = self.brain.act(
                inputs=obs,
                rnn_hxs=recurrent_hidden_states,
                masks=masks,
                deterministic=deterministic,
            )
        return self.action

    def observe(self, obs, reward, done, infos):
        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])

        '''store experiences'''
        self.rollouts.insert(obs, self.recurrent_hidden_states, self.action,
                             self.action_log_prob, self.value, reward, masks)

        '''log'''
        self.episode_scaler_summary.at_step({
            'raw': reward[0].squeeze().item(),
            'len': 1.0,
        })
        if done[0]:
            self.episode_scaler_summary.at_done()
            self.test_obs_at_done()

        self.step_i += 1

    def test_obs_at_act(self, obs):
        if self.test_obs:
            # [thread, stack, H, W] --> (T, H, W)
            obs_frame = obs[0, -1, :, :].unsqueeze(0)
            if self.obs_video is None:
                self.obs_video = obs_frame
            else:
                self.obs_video = torch.cat([self.obs_video, obs_frame], 0)

    def test_obs_at_done(self):
        if self.test_obs:
            self.test_obs = False
            # (T, H, W) --> vid_tensor: :math:`(N, T, C, H, W)`.
            self.obs_video = self.obs_video.unsqueeze(0).unsqueeze(2)
            self.tf_summary.add_video(
                'test_obs/agent_{}'.format(self.id),
                self.obs_video,
            )
            print('# INFO: [Agent {}] test obs done'.format(self.id))

    def experience_not_enough(self):
        return (self.step_i < self.num_steps)

    def after_rollout(self):
        if not self.experience_not_enough():
            self.step_i = 0

    def update(self):
        '''prepare for update'''
        with torch.no_grad():
            obs, recurrent_hidden_states, masks = self.rollouts.get_policy_inputs(
                -1)
            next_value = self.brain.get_value(
                inputs=obs,
                rnn_hxs=recurrent_hidden_states,
                masks=masks,
            ).detach()
        self.rollouts.compute_returns(
            next_value, self.use_gae, self.gamma, self.tau)
        '''update'''
        value_loss, action_loss, dist_entropy = self.trainer.update(
            self.rollouts)

        self.rollouts.after_update()

        '''log info by print'''
        if self.update_i % self.log_interval == 0:
            self.log('learning')

        '''vis curves'''
        if self.vis and self.update_i % self.vis_interval == 0:
            self.vis_curves('learning')

    def get_num_trained_frames(self):
        return self.update_i * self.num_processes * self.num_steps

    def to_print_str(self, mode):
        print_str = "# INFO: [Agent {}][{}]".format(
            self.id,
            mode,
        )
        if mode in ['learning']:
            end = time.time()
            FPS = ((self.get_num_trained_frames() + self.num_processes * self.num_steps) -
                   self.num_trained_frames_start) / (time.time() - self.time_start)
            print_str += "[{}/{}][F-{}][FPS {}][Remain {:.2f} hrs]".format(
                self.update_i, self.num_updates,
                self.get_num_trained_frames(),
                int(FPS),
                ((self.num_env_steps - self.get_num_trained_frames()) / FPS / 60.0 / 60.0),
            )

        return print_str

    def log(self, mode):
        print_str = ''
        print_str += self.to_print_str(mode)
        print_str += self.episode_scaler_summary.to_print_str()
        print(print_str)

    def vis_curves(self, mode):
        if self.episode_scaler_summary.get_length() > 0:
            tmp = self.episode_scaler_summary.to_recent()
            for key in tmp.keys():
                self.tf_summary.add_scalar(
                    '{}/{}'.format(mode, key),
                    tmp[key],
                    self.get_num_trained_frames(),
                )

    def store(self):
        try:
            checkpoint = self.get_num_trained_frames()
            self.store_to_checkpoint(checkpoint)
            self.current_checkpoint_by_frame = checkpoint
            print('# INFO: [Agent {}][Store to checkpoint {} ok]'.format(
                self.id, checkpoint))

        except Exception as e:
            print('# WARNING: [Agent {}][Store failed: {}]'.format(self.id, e))

    def restore(self, principle='recent'):
        '''
            principle:
                recent: the most recent checkpoint,
                uniform: uniformly sampled from all historical checkpoints
                XX_th: the XX-th checkpoint
        '''
        try:
            possible_checkpoints = self.get_possible_checkpoints()
            checkpoint = self.get_checkpoint(possible_checkpoints, principle)
            self.restore_from_checkpoint(checkpoint)
            self.current_checkpoint_by_frame = checkpoint
            print('# INFO: [Agent {}][Restore checkpoint {} ok from {} possible checkpoints, following principle of {}]'.format(
                self.id, checkpoint, len(possible_checkpoints), principle))

        except Exception as e:
            print(
                '# WARNING: [Agent {}][Restore failed: {}]'.format(self.id, e))

    def store_to_checkpoint(self, checkpoint):
        import copy
        from .utils import get_vec_normalize
        # A really ugly way to save a model to CPU
        save_model = self.brain
        if self.cuda:
            save_model = copy.deepcopy(self.brain).cpu()
        save_model = [save_model,
                      getattr(get_vec_normalize(self.envs), 'ob_rms', None)]
        torch.save(save_model, os.path.join(self.log_dir, 'agent_{}'.format(
            checkpoint,
        ) + ".pt"))
        np.save(
            os.path.join(self.log_dir, "update_i.npy"),
            np.array([self.update_i]),
        )

    def restore_from_checkpoint(self, checkpoint):
        self.brain, ob_rms = torch.load(os.path.join(self.log_dir, 'agent_{}.pt'.format(
            checkpoint,
        )))
        if self.cuda:
            self.brain = self.brain.cuda()
        self.envs.ob_rms = ob_rms
        self.update_i = np.load(
            os.path.join(self.log_dir, "update_i.npy"),
        )[0]

    def get_possible_checkpoints(self):
        '''get possible_checkpoints'''
        import glob
        import os
        possible_checkpoints = []
        for file in glob.glob(os.path.join(self.log_dir, 'agent_*.pt')):
            possible_checkpoints += [int(file.split(self.log_dir)
                                         [1].split('.pt')[0].split('agent_')[1])]
        possible_checkpoints = np.asarray(possible_checkpoints)
        possible_checkpoints.sort()
        return possible_checkpoints

    def get_checkpoint(self, possible_checkpoints, principle):
        '''
            recent
            uniform
            {}_th, according to time, ranking
            {}_frame, according to frame
        '''

        if principle in ['recent']:
            return possible_checkpoints[-1]

        elif principle in ['uniform']:
            num_possible_checkpoints = possible_checkpoints.shape[0]
            p = [1.0 / num_possible_checkpoints] * num_possible_checkpoints
            return np.random.choice(possible_checkpoints, p=p)

        elif 'th' in principle:
            return possible_checkpoints[int(principle.split('_th')[0])]

        elif 'frame' in principle:
            return int(principle.split('_frame')[0])
